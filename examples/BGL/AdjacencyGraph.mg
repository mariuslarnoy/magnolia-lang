package examples.BGL.AdjacencyGraph imports examples.BGL.Utils;

// An adjacency graph lets us extract a container of its vertices, and for
// each vertex, the vertices that are adjacent to it.
concept AdjacencyGraph = {
    use ContainerFilter[ Container => VertexContainer
                       , Element => Vertex
                       , member => containsVertex
                       ];

    use ContainerFilter[ Container => AdjacencyVertexContainer
                       , Element => Vertex
                       , member => containsVertex
                       ];

    type Graph;
    function vertices(g: Graph): VertexContainer;
    function adjacentVertices(g: Graph, v: Vertex): AdjacencyVertexContainer
        guard containsVertex(vertices(g), v);

    axiom adjacentVerticesAreVertices(g: Graph, v: Vertex, adj_v: Vertex) {
        assert containsVertex(adjacentVertices(g, v), adj_v) =>
               containsVertex(vertices(g), adj_v);
    }
}

// An undirected adjacency graph is an adjacency graph where each edge is
// symmetric.
concept UndirectedAdjacencyGraph = {
    use AdjacencyGraph;

    axiom adjacencyIsSymmetric(g: Graph, v: Vertex, adj_v: Vertex) {
        assert containsVertex(adjacentVertices(g, v), adj_v) =>
               containsVertex(adjacentVertices(g, adj_v), v);
    }
}

external DFSVisit_external = {
    type Vertex;
    type Graph;
    type VertexContainer;
    type AdjacencyVertexContainer;

    predicate containsVertex(c: VertexContainer, e: Vertex);
    predicate containsVertex(c: AdjacencyVertexContainer, e: Vertex);
    function adjacentVertices (g : Graph, v : Vertex): AdjacencyVertexContainer;
    type State;
    type VertexStack;
    type VertexSet;
    function empty(): State;
    function empty(): VertexStack;
    function empty(): VertexSet;
    predicate isEmpty(s: State);
    predicate isEmpty(v: VertexStack);
    predicate isEmpty(v: VertexSet);
    function pushAll(s: VertexStack, e: AdjacencyVertexContainer): VertexStack;

    predicate isVisited(c: VertexSet, e: Vertex);
    function mkEmptySet(): VertexSet;
    function mkEmptyStack(): VertexStack;
    function mkTuple2(e1: VertexStack, e2: VertexSet): Tuple2;

    // TODO: fix guard bug
    function peek(s: VertexStack): Vertex guard !isEmpty(s);
    function pop (s : VertexStack) : VertexStack guard !isEmpty(s);
    function push (s : VertexStack, e : Vertex) : VertexStack;

    function runDFSLoop (d : State, c : Graph) : State;
    function runDFSStep (d : State, c : Graph) : State;
    function vertexStack (c : State) : VertexStack;
    function vertices (g : Graph) : VertexContainer;
    function visitVertex (c : VertexSet, e : Vertex) : VertexSet;
    function visitedVertices (c : State) : VertexSet;

    procedure setE1 (upd c : Tuple2, obs e : VertexStack);
    procedure setE2 (upd c : Tuple2, obs e : VertexSet);
    procedure setVertexStack (upd c : State, obs e : VertexStack);

    procedure setVisitedVertices (upd c : State, obs e : VertexSet);

    type Tuple2;
    function fst(c: Tuple2): VertexStack;
    function snd(c: Tuple2): VertexSet;
}

program DFSVisit = {
    use DFSVisit_external;

    require type Graph;
    require type Vertex;

    require AdjacencyGraph;

    // How do we deal with importing axioms, and the fact that they have
    // different implementations? For now, we just cast to signatures.

    // The issue is that beyond the type of the function calls in the body
    // of certain axioms, everything else is exactly the same. I guess that
    // again, the solution is to use satisfactions rather than axioms.
    require signature(Stack)[ Stack => VertexStack
                            , Element => Vertex
                            , empty => mkEmptyStack
                            ];

    require function pushAll(s: VertexStack, e: AdjacencyVertexContainer)
                : VertexStack;

    require signature(ExtensibleAndEmptiableContainerFilter)
        [ Container => VertexSet
        , Element => Vertex
        , empty => mkEmptySet
        , add => visitVertex
        , member => isVisited
        ];

    // State is essentially a struct containing a set of vertices and a
    // stack of vertices.
    require type State;

    // I would like to rename "empty" to the same function in both of these
    // imports. How could we do this conveniently?
    require signature(EmptiableElementRWContainer)
        [ Container => State
        , Element => VertexSet
        , projection => visitedVertices
        , set => setVisitedVertices
        ];

    require signature(EmptiableElementRWContainer)
        [ Container => State
        , Element => VertexStack
        , projection => vertexStack
        , set => setVertexStack
        ];

    require signature(WhileLoop)
        [ Context => Graph
        , Data => State
        , cond => loopCond
        , repeat => runDFSLoop
        , body => runDFSStep
        ];

    require signature(Tuple2)
        [ E1 => VertexStack
        , E2 => VertexSet
        ];

    procedure DFS_iterative(obs g: Graph, obs v: Vertex) {
        var state : State = empty();
        call setVertexStack(state, push(vertexStack(state), v));

        // TODO: differentiate value block to avoid this.
        var DUMMY_TODO = runDFSLoop(state, g);
    }

    predicate loopCond(state: State, g: Graph) = !isEmpty(vertexStack(state));
    procedure runDFSStep(upd state: State, obs g: Graph) {
        // TODO: split this into declaration + assignment when improving type
        // inference.
        var _vertexStack = vertexStack(state);
        var _visitedVertices = visitedVertices(state);

        var v : Vertex = peek(_vertexStack);
        call setVertexStack(state, pop(_vertexStack));

        // TODO: assignment of variables by adding to prelude and parser (good first issue?)
        var tup =
            if isVisited(_visitedVertices, v)
            then mkTuple2(vertexStack(state), _visitedVertices)
            else mkTuple2( pushAll(vertexStack(state), adjacentVertices(g, v))
                         , visitVertex(_visitedVertices, v)
                         );

        call setVertexStack(state, fst(tup));
        call setVisitedVertices(state, snd(tup));
    }

    // TODO: Compiler bug
    //procedure bug01() {
    //    var v: VertexStack;
    //    assert isEmpty(v);
    //}
}

/*external Tup = {
    require type E1;
    require type E2;
    type Tuple;

    function construct(e1: E1, e2: E2) : Tuple;
    function p1(t: Tuple): E1;
    function p2(t: Tuple): E2;
}

external Tup3 = {
    require type E1;
    require type E2;
    require type E3;
    type Tuple;
  
    function construct(e1: E1, e2: E2, e3: E3) : Tuple;
    function p1(t: Tuple): E1;
    function p2(t: Tuple): E2;
    function p3(t: Tuple): E3;
}

implementation I = {
    use Tup;
    use Tup3;
}

implementation I2 = {
    use Tup;

    require type E3;
    external function p3(t: Tuple): E3;
};*/
