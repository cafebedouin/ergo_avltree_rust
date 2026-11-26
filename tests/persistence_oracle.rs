#![cfg(feature = "persistence")]
//! Differential oracle for a disk-backed `VersionedAVLStorage`: a seeded schedule of operation batches,
//! rollbacks to stored versions and close-and-reopen cycles. After every step the persistent prover's digest
//! and every key's `unauthenticated_lookup` are compared with an in-memory `BatchAVLProver` that replays the
//! same branch of history from genesis. The backend is chosen only in `backend::open` (one per PR).

use bytes::Bytes;
use ergo_avltree_rust::authenticated_tree_ops::AuthenticatedTreeOps;
use ergo_avltree_rust::batch_avl_prover::BatchAVLProver;
use ergo_avltree_rust::batch_node::{AVLTree, Node, NodeHeader};
use ergo_avltree_rust::operation::{ADDigest, KeyValue, Operation};
use ergo_avltree_rust::persistent_batch_avl_prover::PersistentBatchAVLProver;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::collections::BTreeMap;
use std::path::Path;

const KEY_LENGTH: usize = 32;
const VALUE_LENGTH: Option<usize> = Some(8);
const KEEP: usize = 10;
/// Roll back only to versions written within this many updates (a store may prune older ones, abandoned branches included).
const WINDOW: usize = 5;
/// T2b (amendment A1): also compare each batch's proof bytes with the replay's.
const CHECK_PROOFS: bool = true;

mod backend {
    use super::*;
    use ergo_avltree_rust::persistence::RedbAVLStorage;
    /// #12: redb store, resolver from the store (as in tests/persistence_tests.rs).
    pub fn open(path: &Path) -> PersistentBatchAVLProver {
        let storage = RedbAVLStorage::open(path, KEY_LENGTH, VALUE_LENGTH).unwrap();
        let tree = AVLTree::with_resolver(storage.create_resolver(), KEY_LENGTH, VALUE_LENGTH);
        PersistentBatchAVLProver::new(BatchAVLProver::new(tree, true), Box::new(storage), vec![]).unwrap()
    }
}

fn memory_prover() -> BatchAVLProver {
    BatchAVLProver::new(
        AVLTree::new(
            |d: &[u8; 32]| Node::LabelOnly(NodeHeader::new(Some(*d), None)),
            KEY_LENGTH,
            VALUE_LENGTH,
        ),
        true,
    )
}

struct Version {
    digest: ADDigest,
    batch: Vec<Operation>,
    written: usize,
    model: BTreeMap<Bytes, Bytes>,
}

fn replay(branch: &[Version]) -> BatchAVLProver {
    let mut p = memory_prover();
    p.generate_proof();
    for v in &branch[1..] {
        for o in &v.batch {
            p.perform_one_operation(o).unwrap();
        }
        p.generate_proof();
    }
    p
}

fn check(step: usize, what: &str, pp: &PersistentBatchAVLProver, branch: &[Version], probes: &[Bytes]) {
    let want = &branch.last().unwrap();
    assert_eq!(pp.digest(), want.digest, "step {step} ({what}): digest differs from the replay");
    for k in probes {
        assert_eq!(
            pp.unauthenticated_lookup(k),
            want.model.get(k).cloned(),
            "step {step} ({what}): lookup of {} differs",
            base16::encode_lower(&k[..4])
        );
    }
}

fn run(seed: u64, rounds: usize) -> (usize, usize, usize, usize) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("store");
    let mut rng = StdRng::seed_from_u64(seed);
    let pool: Vec<Bytes> = (0..120)
        .map(|_| Bytes::from((0..KEY_LENGTH).map(|_| rng.gen_range(1..255u8)).collect::<Vec<u8>>()))
        .collect();
    let mut pp = Some(backend::open(&path));
    let mut genesis = memory_prover();
    genesis.generate_proof();
    let mut branch = vec![Version {
        digest: pp.as_ref().unwrap().digest(),
        batch: vec![],
        written: 0,
        model: BTreeMap::new(),
    }];
    assert_eq!(branch[0].digest, genesis.digest().unwrap(), "empty-tree digest differs");
    let (mut batches, mut rollbacks, mut reopens, mut abandoned) = (0, 0, 0, 0);
    let mut written = 0usize;
    for step in 0..rounds {
        let r: f64 = rng.gen();
        let eligible: Vec<usize> = (0..branch.len().saturating_sub(1))
            .filter(|&i| written - branch[i].written < WINDOW)
            .collect();
        if r < 0.12 && !eligible.is_empty() {
            // roll back to a recent version on the current branch (not the current one)
            let i = eligible[rng.gen_range(0..eligible.len())];
            let d = branch[i].digest.clone();
            pp.as_mut().unwrap().rollback(&d).unwrap_or_else(|e| panic!("step {step}: rollback to version {i} failed: {e}"));
            branch.truncate(i + 1);
            rollbacks += 1;
            check(step, "rollback", pp.as_ref().unwrap(), &branch, &pool);
        } else if r < 0.18 && !eligible.is_empty() {
            // abandoned cycle (amendment A2): operations, then a rollback with no proof, as for a block that fails
            let i = eligible[rng.gen_range(0..eligible.len())];
            let model = &branch.last().unwrap().model;
            for _ in 0..rng.gen_range(1..5) {
                let k = pool[rng.gen_range(0..pool.len())].clone();
                let o = if model.contains_key(&k) {
                    Operation::Remove(k)
                } else {
                    Operation::Insert(KeyValue { key: k, value: Bytes::from(vec![7u8; 8]) })
                };
                let _ = pp.as_mut().unwrap().perform_one_operation(&o);
            }
            let d = branch[i].digest.clone();
            pp.as_mut().unwrap().rollback(&d).unwrap_or_else(|e| panic!("step {step}: abandoned-cycle rollback failed: {e}"));
            branch.truncate(i + 1);
            abandoned += 1;
            check(step, "abandoned cycle", pp.as_ref().unwrap(), &branch, &pool);
        } else if r < 0.26 {
            // close and reopen from disk
            drop(pp.take());
            pp = Some(backend::open(&path));
            reopens += 1;
            check(step, "reopen", pp.as_ref().unwrap(), &branch, &pool);
        } else {
            let mut model = branch.last().unwrap().model.clone();
            let mut batch = vec![];
            for _ in 0..rng.gen_range(1..8) {
                let k = pool[rng.gen_range(0..pool.len())].clone();
                let v = Bytes::from(rng.gen::<u64>().to_be_bytes().to_vec());
                let present = model.contains_key(&k);
                let o = match (present, rng.gen_range(0..4)) {
                    (false, _) => Operation::Insert(KeyValue { key: k.clone(), value: v.clone() }),
                    (true, 0) => Operation::Remove(k.clone()),
                    (true, 1) => Operation::Lookup(k.clone()),
                    (true, _) => Operation::Update(KeyValue { key: k.clone(), value: v.clone() }),
                };
                match &o {
                    Operation::Insert(kv) | Operation::Update(kv) => {
                        model.insert(kv.key.clone(), kv.value.clone());
                    }
                    Operation::Remove(k) => {
                        model.remove(k);
                    }
                    _ => {}
                }
                pp.as_mut().unwrap().perform_one_operation(&o).unwrap_or_else(|e| panic!("step {step}: op failed: {e}"));
                batch.push(o);
            }
            let proof = pp.as_mut().unwrap().generate_proof_and_update_storage(vec![]).unwrap_or_else(|e| panic!("step {step}: update failed: {e}"));
            let mut next = replay(&branch);
            for o in &batch {
                next.perform_one_operation(o).unwrap();
            }
            let want_proof = next.generate_proof();
            if CHECK_PROOFS {
                assert_eq!(
                    base16::encode_lower(&proof),
                    base16::encode_lower(&want_proof),
                    "step {step} (batch after {} rollbacks, {} reopens): proof bytes differ from the replay ({} vs {} bytes)",
                    rollbacks, reopens, proof.len(), want_proof.len()
                );
            }
            written += 1;
            branch.push(Version { digest: next.digest().unwrap(), batch, written, model });
            batches += 1;
            check(step, "batch", pp.as_ref().unwrap(), &branch, &pool);
        }
    }
    (batches, rollbacks, reopens, abandoned)
}

#[test]
fn persistent_prover_matches_in_memory_replay() {
    for seed in [1u64, 2, 3, 4, 5] {
        let (b, r, o, a) = run(seed, 300);
        println!("seed {seed}: 300 steps, {b} batches, {r} rollbacks, {a} abandoned cycles, {o} reopens: all agree");
    }
}
