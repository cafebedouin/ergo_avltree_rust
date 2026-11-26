#![cfg(feature = "disk-persistence")]
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

mod backend {
    use super::*;
    use ergo_avltree_rust::storage::DiskBackedAVLStorage;
    /// #9: RocksDB store keeping KEEP versions, its global resolver (as in tests/disk_persistence_test.rs).
    pub fn open(path: &Path) -> PersistentBatchAVLProver {
        let storage = DiskBackedAVLStorage::new(path, KEY_LENGTH, VALUE_LENGTH, KEEP).unwrap();
        let tree = AVLTree::new(DiskBackedAVLStorage::get_resolver(), KEY_LENGTH, VALUE_LENGTH);
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


fn fill(pp: &mut PersistentBatchAVLProver, tag: u8, n: u8) -> ADDigest {
    for i in 0..n {
        let k = Bytes::from(vec![tag, i, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]);
        pp.perform_one_operation(&Operation::Insert(KeyValue { key: k, value: Bytes::from(vec![tag; 8]) })).unwrap();
    }
    pp.generate_proof_and_update_storage(vec![]).unwrap();
    pp.digest()
}

fn key(tag: u8, i: u8) -> Bytes {
    Bytes::from(vec![tag, i, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30])
}

/// Two stores open in one process: A is written, B is opened and written, then A must still read its own nodes
/// from disk (a rollback to its own version forces node loads), and still work after B is dropped.
#[test]
fn two_open_stores_do_not_share_state() {
    let (da, db) = (tempfile::tempdir().unwrap(), tempfile::tempdir().unwrap());
    let mut a = backend::open(&da.path().join("a"));
    let a1 = fill(&mut a, 0xa0, 40);
    let _a2 = fill(&mut a, 0xa1, 10);
    let mut b = backend::open(&db.path().join("b"));
    let _b1 = fill(&mut b, 0xb0, 40);
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        a.rollback(&a1).map_err(|e| e.to_string())?;
        let hit = (0..40).filter(|&i| a.unauthenticated_lookup(&key(0xa0, i)).is_some()).count();
        let leak = (0..40).filter(|&i| a.unauthenticated_lookup(&key(0xb0, i)).is_some()).count();
        Ok::<_, String>((a.digest() == a1, hit, leak))
    }));
    println!("A rolled back to its own version with B open: {r:?} (want Ok((true, 40, 0)))");
    drop(b);
    let r2 = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let a3 = fill(&mut a, 0xa2, 5);
        let old = (0..40).filter(|&i| a.unauthenticated_lookup(&key(0xa0, i)).is_some()).count();
        let new = (0..5).filter(|&i| a.unauthenticated_lookup(&key(0xa2, i)).is_some()).count();
        Ok::<_, String>((a.digest() == a3, old, new))
    }));
    println!("A written and read after B dropped: {r2:?} (want Ok((true, 40, 5)))");
    assert!(matches!(r, Ok(Ok((true, 40, 0)))), "isolation with B open");
    assert!(matches!(r2, Ok(Ok((true, 40, 5)))), "isolation after B dropped");
}
