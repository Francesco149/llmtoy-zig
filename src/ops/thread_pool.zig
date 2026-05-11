/// Persistent thread pool: workers sleep on a condition variable when idle
/// and wake to drain a bounded work queue. No Thread.spawn overhead per call.
///
/// Usage:
///   const pool = try ThreadPool.init(allocator, n_threads, io);
///   defer pool.deinit(allocator);
///   pool.submit(myFunc, &myCtx);  // can call N times
///   pool.wait();                  // block until all submitted jobs finish
const std = @import("std");

pub const Func = *const fn (*anyopaque) void;

const Entry = struct { func: Func, ctx: *anyopaque };

pub const ThreadPool = struct {
    io:      std.Io,
    threads: []std.Thread,
    queue:   []Entry,
    head:    usize = 0,    // producer advances (absolute index)
    tail:    usize = 0,    // workers advance   (absolute index)
    pending: usize = 0,    // submitted but not yet complete

    mutex:      std.Io.Mutex     = std.Io.Mutex.init,
    work_ready: std.Io.Condition = std.Io.Condition.init,
    all_done:   std.Io.Condition = std.Io.Condition.init,
    not_full:   std.Io.Condition = std.Io.Condition.init,
    shutdown:   bool = false,

    pub fn init(allocator: std.mem.Allocator, n: usize, io: std.Io) !*ThreadPool {
        const pool = try allocator.create(ThreadPool);
        errdefer allocator.destroy(pool);

        const threads = try allocator.alloc(std.Thread, n);
        errdefer allocator.free(threads);

        // Queue capacity: at least 4× threads so the caller can submit a full
        // batch before any worker has consumed a job.
        const queue = try allocator.alloc(Entry, @max(n * 4, 64));
        errdefer allocator.free(queue);

        pool.* = .{ .io = io, .threads = threads, .queue = queue };

        var spawned: usize = 0;
        errdefer {
            pool.mutex.lockUncancelable(io);
            pool.shutdown = true;
            pool.work_ready.broadcast(io);
            pool.mutex.unlock(io);
            for (threads[0..spawned]) |t| t.join();
        }
        for (threads) |*t| {
            t.* = try std.Thread.spawn(.{}, workerLoop, .{pool});
            spawned += 1;
        }
        return pool;
    }

    pub fn deinit(pool: *ThreadPool, allocator: std.mem.Allocator) void {
        pool.mutex.lockUncancelable(pool.io);
        pool.shutdown = true;
        pool.work_ready.broadcast(pool.io);
        pool.mutex.unlock(pool.io);
        for (pool.threads) |t| t.join();
        allocator.free(pool.threads);
        allocator.free(pool.queue);
        allocator.destroy(pool);
    }

    /// Enqueue one job. The job's context pointer must remain valid until
    /// the matching `wait()` returns.
    pub fn submit(pool: *ThreadPool, func: Func, ctx: *anyopaque) void {
        pool.mutex.lockUncancelable(pool.io);
        defer pool.mutex.unlock(pool.io);
        const cap = pool.queue.len;
        while (pool.head - pool.tail >= cap) pool.not_full.waitUncancelable(pool.io, &pool.mutex);
        pool.queue[pool.head % cap] = .{ .func = func, .ctx = ctx };
        pool.head += 1;
        pool.pending += 1;
        pool.work_ready.signal(pool.io);
    }

    /// Block until all jobs submitted since the last wait() have completed.
    pub fn wait(pool: *ThreadPool) void {
        pool.mutex.lockUncancelable(pool.io);
        defer pool.mutex.unlock(pool.io);
        while (pool.pending > 0) pool.all_done.waitUncancelable(pool.io, &pool.mutex);
    }

    fn workerLoop(pool: *ThreadPool) void {
        pool.mutex.lockUncancelable(pool.io);
        while (true) {
            while (pool.head == pool.tail and !pool.shutdown) {
                pool.work_ready.waitUncancelable(pool.io, &pool.mutex);
            }
            if (pool.shutdown and pool.head == pool.tail) {
                pool.mutex.unlock(pool.io);
                return;
            }
            const cap   = pool.queue.len;
            const entry = pool.queue[pool.tail % cap];
            pool.tail += 1;
            pool.not_full.signal(pool.io);
            pool.mutex.unlock(pool.io);

            entry.func(entry.ctx); // execute without holding the lock

            pool.mutex.lockUncancelable(pool.io);
            pool.pending -= 1;
            if (pool.pending == 0) pool.all_done.signal(pool.io);
        }
    }
};

// ── tests ─────────────────────────────────────────────────────────────────────

test "ThreadPool: parallel atomic increments" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const N = 100;
    const pool = try ThreadPool.init(allocator, 4, io);
    defer pool.deinit(allocator);

    const Ctx = struct {
        counter: *std.atomic.Value(usize),
        fn run(ctx: *anyopaque) void {
            const self: *@This() = @ptrCast(@alignCast(ctx));
            _ = self.counter.fetchAdd(1, .monotonic);
        }
    };

    var counter = std.atomic.Value(usize).init(0);
    var ctxs: [N]Ctx = undefined;
    for (&ctxs) |*c| c.* = .{ .counter = &counter };

    for (&ctxs) |*c| pool.submit(Ctx.run, c);
    pool.wait();

    try std.testing.expectEqual(@as(usize, N), counter.load(.monotonic));
}

test "ThreadPool: results match serial" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const pool = try ThreadPool.init(allocator, 4, io);
    defer pool.deinit(allocator);

    const N = 32;
    var inputs:  [N]u32 = undefined;
    var outputs: [N]u32 = undefined;
    for (&inputs, 0..) |*v, i| v.* = @intCast(i);

    const Ctx = struct {
        in: *const u32,
        out: *u32,
        fn run(ctx: *anyopaque) void {
            const self: *@This() = @ptrCast(@alignCast(ctx));
            self.out.* = self.in.* * self.in.*;
        }
    };
    var ctxs: [N]Ctx = undefined;
    for (&ctxs, &inputs, &outputs) |*c, *in, *out| c.* = .{ .in = in, .out = out };

    for (&ctxs) |*c| pool.submit(Ctx.run, c);
    pool.wait();

    for (outputs, 0..) |v, i| try std.testing.expectEqual(@as(u32, @intCast(i * i)), v);
}
