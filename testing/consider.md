11/3

Consider the Hamiltonian but as g -> 0. This seems similar to hopfield/Ising Model and possibly brain criticality.

X gate reverses |0> and 1
H gates make it into 0.5 sate which is randomness



[X] Highest weight in layer be addressed [weight[highest] - omega], all other + [omega + (len(highest) - 1)) (look into np.argmax())

11/4

[ ] The issue seems to be that I am just distributing the multiplication among the input values. What I need to be doing is distributing along the output nodes. So switch the dimension.

[ ] I may need to change the weights are created as well. The distribution should be on output and not on input

>>>>>> Abhijit Chakraborty
Classical Information
    Coin toss : prob(heads) = 3/4
                prob(tails) = 1/4
                Total prob = 1
    
    Prob vector = (3/4, 1/4) or 3/4(1, 0) + 1/4(0, 1)
                    3/4 | heads > + 1/4 | tails> 
    
    observer: The coin is in a specific state either heads or tails

Classical Operations
    Tails -> Heads          Heads -> Tails  (i.e., rotating coins)

    | 0, 1 | <- Matrix of flipping a coin
    | 1, 0 |

    | 0, 1 | | 1 |   | 0 |
    | 1, 0 | | 0 | = | 1 | Turns vector head into tails

    | 0, 1 | | 0 |   | 1 |
    | 1, 0 | | 1 | = | 0 | Turns vector tails into heads

    "Bit flip" tails -> heads, heads -> tails aka Matrix operations / gates
    This specific matrix is a "NOT" gate

    All entries are real positive numbers
    All columns add up to 1 <- Probability always sums up to 1

Quantum Information
    A state vector contains probability amplitudes.
        prob = | prob amplitude |^2
    The entries of the state vector can be complex numbers (This is why the abs value before square)

    Classical (3/4, 1/4)        Quantum (1 + 2/sqrt(3), 1/sqrt(3))

    "Coin Toss" : two level system (Atom has two levels e.g. electron level switch) (up or down)
        Up (1, 0), Down (0, 1)

        | \psi > "ket" <- Donates quantum amplitude

        | \psi > = 1 / Sqrt(2) + (1, 0) + 1 / Sqrt(2) (0, 1)

        1 / Sqrt(2) are probability amplitudes

        Prob of getting (1, 0) = (1 / sqrt(2) ) ^ 2 = 1/2
        Prob of getting (0, 1) = (1 / sqrt(2) ) ^ 2 = 1/2

        | \psi >  1 + 2 / sqrt(3) (1, 0) + 1 / sqrt(3) (0, 1)

        prob of (1, 0) => | 1 + 2 / sqrt(3) | = (1 + 2)(1 + 2) / 3 = 2/3
        prob of (0, 1) => | 1 / sqrt(3) | = 1/3

        2/3 + 1/3 = 1

        | \psi > = 1/ sqrt(2) (1, 0) + 1/sqrt(2) (0, 1) "super position"


Classical



Quantum
 | \psi > = \alpha (1, 0) + \beta (1, 1)

          = \alpha (1, 0) + beta(1, 0) + \beta (0, 1)

          = (\alpha + \beta) + (1, 0) + \beta(0, 1)

prob(1, 0) = | \alpha + \beta | ^2

           = | \alpha | ^ 2 + | \beta | ^ 2 + 2 \alpha \beta

2 \alpha \beta <- Interference

Superposition
    | \psi > = 1 / sqrt(2) [(1, 0) + (0, 1)]

    | \psi' > = 1 / sqrt(2) [(1, 0) - (0, 1)]

    | \psi > -> prob (1, 0) = 1/2       |  | \psi' > prob of (1, 0) = 1/2
                prob (0, 1) = 1/2       |  | \psi' > prob of (0, 1) = (- 1/ sqrt(2))^2 
                                           | \psi' > prob of (0, 1) = 1/2

"What is different for the neg sign"

Quantum Operations: 
x = [0, 1]
    [1, 0]

x(1, 0) = (0, 1)
x(0, 1) = (1, 0)

                [1,  1]
H = 1 / sqrt(2) [1, -1]

H(1, 0) = 1/sqrt (1, 1) <- Superposition state
        = 1 sqrt(2) [ (1, 0) + (0, 1)]

H(1, 0) turns a head position into a superposition on the "up" or "down" state

| \psi  > 1 / sqrt(2) [(1, 0) + (0, 1)]

| \psi' > 1 / sqrt(2) [(1, 0) - (0, 1)]

                         [1,  1]
H | \psi > = 1 / sqrt(2) [1, -1] 1/sqrt(1,1)
           = (1, 0)

H | \psi' > = 1 / sqrt(2) [1, -1] 1/sqrt(1,-1)
           = (0, 1)

The sign is important (neg sign) determines the phase (i.e., up or down)

We want this quantum operation to have these probabilities

Quantum Operations:
u = Quantum Operation

u | \psi > | \psi_1 > = \alpha (1, 0) + \beta (0, 1)
                      = \alpha^2 + \beta^2 = 1


| \psi > = \alpha_0 (1, 0) + \beta+0(0, 1)

| \alpha_0 | ^ 2 + | \beta_0 | ^ 2 = 1 -> | \alpha_1 | ^ 2 + | \beta_1 | ^ 2 = 1

Conservation of prbabilities ->
    u should have property u u\dagger = u\dagger u = 1
    u = matrix      u\dagger = (u^*)^T

    (1, 1 + \imaginary^\dot)
    (1 - 2^dot, 0)
      
      (1, - 1- 2^dot)
u^* = (1 + \imaginary^\dot, 0)

                       (1, 1 + \imaginary)
u^\dagger = (u^*)^T =  (1 - \imaginary, 0)

Bloch Sphere
| \psi > = \alpha (1, 0) + \beta (0, 1) = | \alpha |  ^ 2 + | \beta | ^ 2 = 1

= cos (\omega / 2) (1, 0) + e^(\imaginary\phi) sin (\omega/2) (0, 1)

cos(\omega/2)]^2 + | e^(\imagarny\phi) sin(\omeage/2)|^2 = cos^2(\omega/2) + sin^2(\omega/2) = 1

Given a sphere the answer/probability is a position on it's surface. Given the | \psi > = (\omega, \phi) are just rotations on its surface.

No matter what operation you do, you only move along the surface of the sphere.
u -> unitarty

Called the "Bloch Sphere" named after Dr. Bloch a quantum physicist

Measurement collapses a superposition to one of the possible outcomes.

Z -> measurement (1, 0) or (0, 1)


RECAP
Quantum states can have complex probability amplitutes | \psi > = \alpha(1, 0) + \beta(0, 1)
                \alpha and \beta can be imaganry as long | \alpha |^2 + | \beta |^2 = 1
Prob by taking amplitude square | \alpha |^2 or | \beta |^2

Measure a state it collapses the superposition
Operation on quantum states are unitary uu^\dagger = u^\dagger+u = 1

Entanglement
System has 2 particles 

| 0 > = (1, 0)
| 1 > = (0, 1)

For system A {|0_A >, | 1_A > } B -> {|0_B>, | 1_B>}

A + B possible states {|0_A 0_B>, |0_A 1_B>, |1_A 0_B>, |1_A 1_B>,}

| \psi >_A+B = \alpha_1 | 0_A0_B> + \alpha_2 | 1_A0_B> + \alpha_3 | 0_A1_B> + \alpha_4 | 1_A1_B>

| \psi >_A+B = | \psi A > \tensorProduct | \psi_B >  

| \psi_A > = (\alpha_1, \beta_1)
| \psi_A > = (\alpha_2, \beta_2)

( \alpha_1, \beta_1) \tensorProduct (\alpha_2, \beta_2) = All possibilities

These systems become entangled. This leads to the power of quantum computation.



>>>>>> Clarice Aiello
[ ] Ask about actin and microtubules in respect to genomic instability (e.g polyploidy and repositioning found in cancer)

>>>>>>> Pablo Lopez Duque
Introduction to Quantum Computing and Qiskit

# X GATE
Is a NOT gate or Bit Flip Gate
|0> -> |1>
|1> -> |0>
\alpha|0> + \beta|1> -X-> \alpha|1> + \beta|0>

      [0, 1]
NOT = [1, 0]

# H GATE
Puts you into super position
              []
h = 1/sqrt(2) []

# CNOT GATE
Controlled not gate

|0>|0> -> |0>|0>
|0>|1> -> !1>|1>
|1>|0> -> |1>|0>
|1>|1> -> |0>|1>

        [ 1 0 0 0 ]
        [ 0 1 0 0 ]
        [ 0 0 0 1 ]
CNOT=   [ 0 0 1 0 ]


# Entanglement -> Bell States
If we combine a Hadamard gate and a CNOT gate, we can get entangled states

>>>>>> Fall Festival Qiskit Challenge
3 Challenge Books
2 tie-breaker notebooks

Free IBM swag items

Weightage:
- 75% accuracy
- 25% alloted to time

email to uhqiskitfest@outlook.com