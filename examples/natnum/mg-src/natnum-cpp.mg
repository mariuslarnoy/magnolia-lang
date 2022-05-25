package examples.natnum.mg-src.natnum-cpp;

concept Semigroup = {

    type S;

    function bop(s1: S, s2: S): S;

    axiom associative(s1: S, s2: S, s3: S) {
        assert bop(bop(s1, s2), s3) == bop(s1, bop(s2, s3));
    }
}

concept AbelianMonoid = {

    // include Semigroup, rename type S to M
    use Semigroup[S => M];

    function identity(): M;

    axiom idAxiom(m: M) {
        assert bop(identity(), m) == m;
        assert bop(m, identity()) == m;
    }
    axiom commutative(m1: M, m2: M) {
        assert bop(m1, m2) == bop(m2, m1);
    }
}

concept Semiring = {
    // Gives us + and 0
    use AbelianMonoid[bop => _+_, identity => zero,
                      associative => associative0,
                      commutative => commutative0,
                      idAxiom => idAxiom0];
    // Gives us * and 1
    use AbelianMonoid[bop => _*_, identity => one];

    // Multiplication distributes over addition
    axiom multDistribution(m1: M, m2: M, m3: M) {
        assert m1 * (m2 + m3) == (m1 * m2) + (m1 * m3);
        assert (m1 + m2) * m3 == (m1 * m3) + (m2 * m3);
    }

    // Annihilation of mult by zero
    axiom multAnnihilation(m: M) {
        assert m * zero() == zero();
        assert zero() * m == zero();
    }
}

implementation ExternalNat = external C++ base.nat {
    type Nat;

    function zero(): Nat;
    function one(): Nat;

    function add(a: Nat, b: Nat): Nat;
    function mul(a: Nat, b: Nat): Nat;
}

program NaturalNumbers = {
    use ExternalNat;
}

satisfaction NaturalNumbersModelsSemiring = NaturalNumbers
    models Semiring[M => Nat,
                 zero => zero,
                  one => one,
                  _+_ => add,
                  _*_ => mul];