package examples.BGL.Utils;

// A container with an associated element type and a predicate which, for
// a given element, gives the information of whether it is in the container.
concept ContainerFilter = {
    type Container;
    type Element;

    predicate member(c: Container, e: Element);
};

concept EmptiableType = {
    type T;

    function empty() : T;
    predicate isEmpty(t: T);

    axiom emptyIsEmpty() {
        assert isEmpty(empty());
    }
}

concept Stack = {
    type Stack;
    type Element;

    use EmptiableType[T => Stack];

    function pop(s: Stack) : Stack guard !isEmpty(s);
    function peek(s: Stack) : Element guard !isEmpty(s);
    function push(s: Stack, e: Element) : Stack;

    axiom pushedStackIsNonEmpty(s: Stack, e: Element) {
        assert !isEmpty(push(s, e));
    }

    axiom popIsLeftInverseOfPush(s: Stack, e: Element) {
        assert pop(push(s, e)) == s;
    }

    axiom peekElementIsPushedElement(s: Stack, e: Element) {
        assert peek(push(s, e)) == e;
    }
}

concept WhileLoop = {
    type Context; // Env
    type Data; // Mutable data

    predicate cond(d: Data, c: Context);
    function body(d: Data, c: Context) : Data;
    function repeat(d: Data, c: Context) : Data;

    axiom whileRepetitions(d: Data, c: Context) {
        assert cond(d, c) => repeat(d, c) == repeat(body(d, c), c);
        assert !cond(d, c) => repeat(d, c) == d;
    }
}

// Name could be changed. One container type with several element types and/or
// projections is equivalent to a readable and writable product type.
concept RWContainer = {
    type Container;
    type Element;

    function projection(c: Container) : Element;
    procedure set(upd c: Container, obs e: Element);

    axiom setUpdatesContainer(c: Container, e: Element) {
        call set(c, e);
        assert projection(c) == e;
    }
}

concept EmptiableElementRWContainer = {
    use RWContainer[ Container => Container
                   , Element => Element
                   ];

    use EmptiableType[T => Element, emptyIsEmpty => emptyElementIsEmpty];
    use EmptiableType[T => Container, emptyIsEmpty => emptyContainerIsEmpty];

    axiom emptyContainerMeansEmptyElements(c: Container) {
        assert isEmpty(c) => isEmpty(projection(c));
    }
}

// Some of what is below can be part of satisfactions... But we can not really
// use them yet.
concept ExtensibleAndEmptiableContainerFilter = {
    use ContainerFilter[ Container => Container
                       , Element => Element
                       , member => member
                       ];

    function add(c: Container, e: Element) : Container;

    use EmptiableType[T => Container];

    axiom emptyContainerContainsNoMember(c: Container, e: Element) {
        assert isEmpty(c) => !member(c, e);
    }

    axiom addMakesMember(c: Container, e: Element) {
        assert member(add(c, e), e);
    }
}

concept Tuple2 = {
    use RWContainer
        [ Container => Tuple2
        , Element => E1
        , projection => fst
        , set => setE1
        ];

    use RWContainer
        [ Container => Tuple2
        , Element => E2
        , projection => snd
        , set => setE2
        ];

    function mkTuple2(e1: E1, e2: E2) : Tuple2;

    axiom projectionsRelationWithConstruction(e1: E1, e2: E2) {
        var tup = mkTuple2(e1, e2);
        assert fst(tup) == e1 && snd(tup) == e2;
    }
}
