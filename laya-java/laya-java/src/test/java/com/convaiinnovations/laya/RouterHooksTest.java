package com.convaiinnovations.laya;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.convaiinnovations.laya.Router.Checkpoint;
import com.convaiinnovations.laya.hooks.Hook;
import com.convaiinnovations.laya.hooks.PredictContext;
import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/**
 * A Router-level hook sees what no agent can: which checkpoint was chosen, and when one was built
 * or dropped. Before this, {@code Router} dispatched nothing at all.
 */
class RouterHooksTest {

    /** Records every event with the model it carried. */
    private static final class Trace implements Hook {

        final List<String> events = new ArrayList<>();

        private void note(String event, PredictContext ctx) {
            events.add(event + ":" + ctx.model());
        }

        @Override
        public void onRoute(PredictContext ctx) {
            note("route", ctx);
        }

        @Override
        public void onLoad(PredictContext ctx) {
            note("load", ctx);
        }

        @Override
        public void onEvict(PredictContext ctx) {
            note("evict", ctx);
        }

        @Override
        public void onPredictStart(PredictContext ctx) {
            note("start", ctx);
        }

        @Override
        public void onPredictEnd(PredictContext ctx) {
            note("end", ctx);
        }
    }

    private static final class StubAgents implements Router.AgentFactory {

        private final Path root;
        int builds;

        StubAgents(Path root) {
            this.root = root;
        }

        @Override
        public Agent create(Checkpoint checkpoint) throws IOException {
            builds++;
            return TinyCheckpoint.agent(root, new TinyCheckpoint.RecordingSession());
        }
    }

    private static Map<String, Question> questions() {
        return Map.of("urgent", Question.noul("Needs a human."));
    }

    @Test
    @DisplayName("route() dispatches on_route once, naming the checkpoint it chose")
    void routeDispatches(@TempDir Path root) throws IOException {
        TinyCheckpoint.write(root, 64, 32);
        Trace trace = new Trace();
        try (Router router = Router.builder().agents(new StubAgents(root)).build()) {
            router.hooks().addHook(trace);
            router.route("hello", questions());
            assertEquals(1, trace.events.size(), trace.events.toString());
            assertTrue(trace.events.get(0).startsWith("route:"), trace.events.toString());
        }
    }

    @Test
    @DisplayName("load() dispatches on_load for a build and stays quiet for a cache hit")
    void loadDispatchesOnlyOnBuild(@TempDir Path root) throws IOException {
        TinyCheckpoint.write(root, 64, 32);
        Trace trace = new Trace();
        StubAgents agents = new StubAgents(root);
        try (Router router = Router.builder().agents(agents).maxLoaded(2).build()) {
            router.hooks().addHook(trace);
            router.load("english");
            assertEquals(List.of("load:english"), trace.events);
            router.load("english");                    // resident: no second build, no second event
            assertEquals(List.of("load:english"), trace.events);
            assertEquals(1, agents.builds);
        }
    }

    @Test
    @DisplayName("an eviction forced by maxLoaded dispatches on_evict for the victim")
    void evictionDispatches(@TempDir Path root) throws IOException {
        TinyCheckpoint.write(root, 64, 32);
        Trace trace = new Trace();
        try (Router router = Router.builder().agents(new StubAgents(root)).maxLoaded(1).build()) {
            router.hooks().addHook(trace);
            router.load("english");
            router.load("multilingual");               // maxLoaded 1, so english must go
            assertTrue(trace.events.contains("evict:english"), trace.events.toString());
            assertEquals(List.of("load:english", "load:multilingual", "evict:english"),
                    trace.events);
        }
    }

    @Test
    @DisplayName("unload dispatches on_evict for what it freed")
    void unloadDispatches(@TempDir Path root) throws IOException {
        TinyCheckpoint.write(root, 64, 32);
        Trace trace = new Trace();
        try (Router router = Router.builder().agents(new StubAgents(root)).build()) {
            router.load("english");
            router.hooks().addHook(trace);             // installed after the load
            assertEquals(List.of(Checkpoint.ENGLISH), router.unload("english"));
            assertEquals(List.of("evict:english"), trace.events);
        }
    }

    @Test
    @DisplayName("predict dispatches one start/end pair for the whole route-and-answer call")
    void predictDispatchesOnePair(@TempDir Path root) throws IOException {
        TinyCheckpoint.write(root, 64, 32);
        Trace trace = new Trace();
        try (Router router = Router.builder().agents(new StubAgents(root)).build()) {
            router.hooks().addHook(trace);
            router.predict("hello", questions());
            assertEquals(1, trace.events.stream().filter(e -> e.startsWith("start:")).count(),
                    trace.events.toString());
            assertEquals(1, trace.events.stream().filter(e -> e.startsWith("end:")).count(),
                    trace.events.toString());
            // on_load lands inside the pair, which is the order the reference dispatches in.
            int start = trace.events.indexOf("start:english");
            int load = trace.events.indexOf("load:english");
            int end = trace.events.indexOf("end:english");
            assertTrue(start >= 0 && load > start && end > load, trace.events.toString());
        }
    }

    @Test
    @DisplayName("a hook that calls back into the router does not deadlock")
    void aReentrantHookDoesNotDeadlock(@TempDir Path root) throws IOException {
        TinyCheckpoint.write(root, 64, 32);
        List<Integer> seen = new ArrayList<>();
        try (Router router = Router.builder().agents(new StubAgents(root)).maxLoaded(2).build()) {
            // Dispatching under the router lock would park this call on itself for ever.
            router.hooks().addHook(new Hook() {
                @Override
                public void onLoad(PredictContext ctx) {
                    seen.add(router.loaded().size());
                }
            });
            router.load("english");
            assertEquals(List.of(1), seen);
        }
    }

    @Test
    @DisplayName("a router with no hooks dispatches nothing and builds no context")
    void noHooksNoDispatch(@TempDir Path root) throws IOException {
        TinyCheckpoint.write(root, 64, 32);
        StubAgents agents = new StubAgents(root);
        try (Router router = Router.builder().agents(agents).build()) {
            assertFalse(router.hooks().hooks().iterator().hasNext());
            router.predict("hello", questions());      // the unhooked fast path still answers
            assertEquals(1, agents.builds);
        }
    }
}
