//! Counts the heap allocations of deferred calls once warm, which must be none: a `Deferred` keeps
//! its outcome, arguments and inputs inline, at the address it is pinned at.
//!
//! File: rust/allocations.rs
//! Author: Ash Vardanian
//! Date: November 18, 2026

use std::alloc::{GlobalAlloc, Layout, System};
use std::future::Future;
use std::pin::pin;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::task::{Context, Poll, Waker};

use uform::{Executor, Model, ModelVerbs};

struct Counting;

static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        unsafe { System.dealloc(pointer, layout) }
    }

    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        unsafe { System.realloc(pointer, layout, size) }
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

/// Polls `future` to completion on this thread with a no-op waker.
fn block_on<F: Future>(future: F) -> F::Output {
    let mut future = pin!(future);
    let mut context = Context::from_waker(Waker::noop());
    loop {
        if let Poll::Ready(output) = future.as_mut().poll(&mut context) {
            return output;
        }
        std::thread::yield_now();
    }
}

#[test]
fn deferred_calls_allocate_nothing_once_warm() {
    let checkpoint = match std::env::var_os("UFORM_TEST_CHECKPOINT") {
        Some(path) => std::fs::read(path).expect("UFORM_TEST_CHECKPOINT names a readable checkpoint"),
        None => vec![7u8; 256],
    };
    let model = Model::load_ram_buffer(&checkpoint).unwrap();
    let executor = Executor::for_ram(&[], None, 1).unwrap();
    let expected = model.describe().unwrap();
    for round in 0..2 {
        let before = ALLOCATIONS.load(Ordering::Relaxed);
        for _ in 0..50 {
            let waited = model.deferred(&executor).describe().unwrap().wait().unwrap();
            let awaited = block_on(model.deferred(&executor).describe().unwrap()).unwrap();
            assert!(waited == expected && awaited == expected);
        }
        let allocations = ALLOCATIONS.load(Ordering::Relaxed) - before;
        if round == 1 {
            assert_eq!(allocations, 0, "100 warm deferred calls allocated");
        }
    }
}
