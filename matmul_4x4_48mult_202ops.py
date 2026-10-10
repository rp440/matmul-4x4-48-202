"""
4x4 matrix multiplication with 48 multiplications and 202 linear operations
(L: 54 add + 2 halvings, R: 54 add + 2 halvings, P: 90 add).

Valid over any ring containing 1/2, including noncommutative entries (matrix blocks).
A, B, C are indexed row-major: a[4*i+j] = A[i][j]. Product k is m_k = L_k * R_k.

Running this file checks all 4096 coefficient identities exactly (Fraction
arithmetic) and 1000 random integer products.
"""
from fractions import Fraction

HALF = Fraction(1, 2)


def multiply_4x4_202(A, B):
    """Return C = A @ B for 4x4 nested lists A, B, using 48 multiplications."""
    a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15 = [x for row in A for x in row]
    b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15 = [x for row in B for x in row]

    # ---- Stage L: 48 linear combinations of A's entries ----
    L_v16 = a0+a3
    L_v17 = a1+L_v16
    L_v18 = a6-a5
    L_v19 = a14-a13
    L_v20 = a1+a13
    L_v21 = L_v20-a9
    L_v22 = L_v20-a5
    L_v23 = a6-a7
    L_v24 = a4+a7
    L_v25 = L_v16+L_v24
    L_v26 = L_v25*HALF
    L_v27 = L_v16-L_v26
    L_v28 = a12+L_v27
    L_v29 = L_v26-a8
    L_v30 = a15+L_v28
    L_v31 = L_v26-L_v18
    L_v32 = a2-a3
    L_v33 = a1+L_v26
    L_v34 = L_v24-L_v18
    L_v36 = a11-L_v29
    L_v40 = L_v21-L_v36
    L_v42 = L_v30+L_v40
    L_v77 = a10+a14
    L_v43 = L_v77*HALF
    L_v44 = a6-L_v43
    L_v45 = a7-L_v44
    L_v50 = L_v22+L_v44
    L_v61 = L_v40-L_v50
    L_v62 = L_v31-L_v61
    L_v64 = L_v33+L_v61
    L_v67 = L_v44+L_v64
    L_v68 = L_v43+L_v62
    L_v70 = a10-L_v68
    L_v72 = L_v45-L_v29
    L_v38 = L_v43-a13
    L_v39 = a10-L_v43
    L_v46 = a2-L_v39
    L_v47 = a3-L_v46
    L_v48 = L_v28+L_v72
    L_v49 = L_v43+L_v48
    L_v51 = L_v50-L_v46
    L_v52 = L_v47+L_v72
    L_v56 = L_v36-L_v52
    L_v59 = L_v26-L_v52
    L_v63 = L_v38+L_v61
    L_v65 = L_v46-L_v64
    L_v66 = L_v30-L_v63
    L_v69 = L_v36+L_v63
    L_v71 = L_v28-L_v47
    L_v41 = L_v44+L_v48
    L_v53 = L_v29+L_v71
    L_v54 = L_v45+L_v71
    L_v55 = L_v30-L_v54
    L_v57 = L_v39-L_v53
    L_v58 = L_v27+L_v54
    L_v60 = L_v46-L_v53
    L_o0 = L_v17
    L_o1 = L_v23
    L_o2 = L_v70
    L_o3 = L_v53
    L_o4 = L_v69
    L_o5 = L_v55
    L_o6 = L_v65
    L_o7 = a11
    L_o8 = L_v31
    L_o9 = a13
    L_o10 = L_v56
    L_o11 = L_v28
    L_o12 = L_v19
    L_o13 = L_v57
    L_o14 = L_v62
    L_o15 = a15
    L_o16 = L_v48
    L_o17 = L_v58
    L_o18 = L_v54
    L_o19 = L_v59
    L_o20 = L_v32
    L_o21 = L_v29
    L_o22 = L_v51
    L_o23 = L_v18
    L_o24 = L_v63
    L_o25 = a1
    L_o26 = L_v41
    L_o27 = L_v33
    L_o28 = L_v34
    L_o29 = L_v60
    L_o30 = a7
    L_o31 = L_v52
    L_o32 = L_v66
    L_o33 = L_v47
    L_o34 = L_v40
    L_o35 = a8
    L_o36 = L_v22
    L_o37 = L_v50
    L_o38 = L_v67
    L_o39 = L_v45
    L_o40 = a12
    L_o41 = a3
    L_o42 = L_v68
    L_o43 = L_v38
    L_o44 = L_v42
    L_o45 = L_v21
    L_o46 = L_v49
    L_o47 = L_v64

    # ---- Stage R: 48 linear combinations of B's entries ----
    R_v16 = b6+b10
    R_v17 = b7+b11
    R_v20 = b11-b8
    R_v26 = b0-b12
    R_v27 = b2-b14
    R_v40 = b14-b6
    R_v68 = b12-b4
    R_v70 = b5-b7
    R_v21 = R_v20-b9
    R_v28 = R_v26+R_v27
    R_v36 = b10+R_v20
    R_v58 = R_v70*HALF
    R_v60 = R_v16-R_v27
    R_v62 = R_v26-b8
    R_v23 = R_v17+R_v58
    R_v25 = b4+R_v58
    R_v29 = R_v28*HALF
    R_v30 = b1+R_v29
    R_v31 = R_v26-R_v29
    R_v33 = R_v25-b12
    R_v78 = b11-R_v23
    R_v32 = b3-R_v31
    R_v37 = R_v62-R_v31
    R_v39 = R_v40+R_v78
    R_v55 = R_v31-R_v60
    R_v72 = b13-R_v30
    R_v38 = R_v78-b6
    R_v44 = R_v21-R_v72
    R_v49 = R_v32-b15
    R_v71 = R_v32+R_v39
    R_v24 = R_v36-R_v38
    R_v48 = R_v33+R_v71
    R_v53 = R_v30+R_v71
    R_v67 = R_v49-R_v44
    R_v74 = R_v23-R_v44
    R_v18 = R_v25-R_v24
    R_v43 = R_v37+R_v74
    R_v45 = R_v24+R_v74
    R_v46 = R_v55-R_v74
    R_v50 = R_v48-R_v49
    R_v57 = R_v53+R_v78
    R_v61 = R_v53-R_v38
    R_v64 = R_v31-R_v48
    R_v69 = R_v48-R_v40
    R_v19 = R_v45-R_v72
    R_v47 = R_v38-R_v46
    R_v52 = R_v57-R_v69
    R_v63 = R_v45-R_v49
    R_v65 = R_v43+R_v58
    R_v66 = R_v25-R_v46
    R_v34 = R_v52-R_v71
    R_v42 = b5-R_v65
    R_v54 = R_v52+R_v72
    R_v56 = R_v29+R_v52
    R_v35 = R_v25+R_v34
    R_v59 = R_v34-R_v58
    R_o0 = R_v35
    R_o1 = R_v19
    R_o2 = R_v36
    R_o3 = R_v37
    R_o4 = R_v40
    R_o5 = b13
    R_o6 = R_v47
    R_o7 = R_v50
    R_o8 = R_v53
    R_o9 = R_v17
    R_o10 = R_v54
    R_o11 = R_v32
    R_o12 = b11
    R_o13 = b8
    R_o14 = R_v24
    R_o15 = b15
    R_o16 = R_v55
    R_o17 = b1
    R_o18 = R_v30
    R_o19 = R_v56
    R_o20 = R_v21
    R_o21 = R_v48
    R_o22 = R_v42
    R_o23 = R_v57
    R_o24 = R_v39
    R_o25 = R_v59
    R_o26 = R_v60
    R_o27 = R_v34
    R_o28 = R_v61
    R_o29 = R_v62
    R_o30 = R_v63
    R_o31 = R_v52
    R_o32 = b14
    R_o33 = R_v44
    R_o34 = R_v33
    R_o35 = R_v64
    R_o36 = R_v65
    R_o37 = R_v43
    R_o38 = R_v66
    R_o39 = R_v45
    R_o40 = b3
    R_o41 = R_v67
    R_o42 = R_v18
    R_o43 = R_v23
    R_o44 = b12
    R_o45 = R_v68
    R_o46 = R_v16
    R_o47 = R_v46

    # ---- The 48 multiplications ----
    m0 = L_o0 * R_o0
    m1 = L_o1 * R_o1
    m2 = L_o2 * R_o2
    m3 = L_o3 * R_o3
    m4 = L_o4 * R_o4
    m5 = L_o5 * R_o5
    m6 = L_o6 * R_o6
    m7 = L_o7 * R_o7
    m8 = L_o8 * R_o8
    m9 = L_o9 * R_o9
    m10 = L_o10 * R_o10
    m11 = L_o11 * R_o11
    m12 = L_o12 * R_o12
    m13 = L_o13 * R_o13
    m14 = L_o14 * R_o14
    m15 = L_o15 * R_o15
    m16 = L_o16 * R_o16
    m17 = L_o17 * R_o17
    m18 = L_o18 * R_o18
    m19 = L_o19 * R_o19
    m20 = L_o20 * R_o20
    m21 = L_o21 * R_o21
    m22 = L_o22 * R_o22
    m23 = L_o23 * R_o23
    m24 = L_o24 * R_o24
    m25 = L_o25 * R_o25
    m26 = L_o26 * R_o26
    m27 = L_o27 * R_o27
    m28 = L_o28 * R_o28
    m29 = L_o29 * R_o29
    m30 = L_o30 * R_o30
    m31 = L_o31 * R_o31
    m32 = L_o32 * R_o32
    m33 = L_o33 * R_o33
    m34 = L_o34 * R_o34
    m35 = L_o35 * R_o35
    m36 = L_o36 * R_o36
    m37 = L_o37 * R_o37
    m38 = L_o38 * R_o38
    m39 = L_o39 * R_o39
    m40 = L_o40 * R_o40
    m41 = L_o41 * R_o41
    m42 = L_o42 * R_o42
    m43 = L_o43 * R_o43
    m44 = L_o44 * R_o44
    m45 = L_o45 * R_o45
    m46 = L_o46 * R_o46
    m47 = L_o47 * R_o47

    # ---- Stage P: assemble the 16 entries of C ----
    P_v48 = m0-m25
    P_v49 = m9+m43
    P_v50 = m18-P_v49
    P_v51 = P_v50-m17
    P_v52 = m27-m25
    P_v53 = m3+P_v52
    P_v54 = P_v53-m13
    P_v55 = P_v51-P_v54
    P_v56 = m21-m35
    P_v57 = P_v56-P_v49
    P_v58 = m19+m31
    P_v59 = m11-m40
    P_v60 = P_v58+P_v59
    P_v61 = P_v57-P_v60
    P_v62 = P_v55-P_v61
    P_v63 = P_v48+P_v62
    P_v64 = m29+P_v53
    P_v65 = P_v63-P_v64
    P_v66 = m37-P_v58
    P_v67 = m22+P_v66
    P_v68 = m23-P_v61
    P_v69 = m8+P_v68
    P_v70 = m9+m12
    P_v71 = P_v69-P_v70
    P_v72 = P_v71-m33
    P_v73 = P_v62-P_v72
    P_v74 = m20+P_v73
    P_v75 = P_v67-P_v74
    P_v76 = m28+P_v68
    P_v77 = P_v60-P_v69
    P_v78 = m16+P_v77
    P_v79 = m46+P_v78
    P_v80 = P_v79-P_v51
    P_v81 = P_v76+P_v80
    P_v82 = m14-P_v59
    P_v83 = P_v80+P_v82
    P_v84 = P_v62+P_v83
    P_v85 = m2+P_v84
    P_v86 = P_v70+P_v85
    P_v87 = P_v81-P_v86
    P_v88 = m6-P_v87
    P_v89 = m47-P_v56
    P_v90 = P_v88-P_v89
    P_v91 = P_v64-P_v67
    P_v92 = P_v90+P_v91
    P_v93 = P_v72-P_v91
    P_v94 = P_v93-m41
    P_v95 = m42+P_v84
    P_v96 = P_v95-P_v63
    P_v97 = m38+P_v96
    P_v98 = m36-P_v66
    P_v99 = m26+P_v78
    P_v100 = P_v98-P_v99
    P_v101 = P_v97+P_v100
    P_v102 = P_v89+P_v101
    P_v103 = m1-P_v80
    P_v104 = P_v52+P_v57
    P_v105 = m39+P_v104
    P_v106 = P_v103+P_v105
    P_v107 = P_v106-P_v98
    P_v108 = P_v81+P_v99
    P_v109 = m30+P_v61
    P_v110 = P_v109-P_v105
    P_v111 = P_v100+P_v110
    P_v112 = m34-P_v77
    P_v113 = m45+P_v112
    P_v114 = P_v95+P_v113
    P_v115 = m10-m19
    P_v116 = P_v115-P_v113
    P_v117 = P_v116-P_v74
    P_v118 = P_v106+P_v117
    P_v119 = m24-P_v104
    P_v120 = m4-P_v119
    P_v121 = P_v118-P_v120
    P_v122 = P_v86+P_v120
    P_v123 = m7-m35
    P_v124 = P_v123-P_v70
    P_v125 = P_v113+P_v124
    P_v126 = P_v125-P_v120
    P_v127 = P_v54+P_v112
    P_v128 = m44+P_v127
    P_v129 = P_v96+P_v128
    P_v130 = m5+m17
    P_v131 = P_v74+P_v130
    P_v132 = P_v106+P_v131
    P_v133 = m32+P_v79
    P_v134 = P_v119+P_v133
    P_v135 = P_v87+P_v134
    P_v136 = m15+m40
    P_v137 = P_v70+P_v136
    P_o0 = P_v65
    P_o1 = P_v75
    P_o2 = P_v92
    P_o3 = P_v94
    P_o4 = P_v102
    P_o5 = P_v107
    P_o6 = P_v108
    P_o7 = P_v111
    P_o8 = P_v114
    P_o9 = P_v121
    P_o10 = P_v122
    P_o11 = P_v126
    P_o12 = P_v129
    P_o13 = P_v132
    P_o14 = P_v135
    P_o15 = P_v137

    c = [P_o0, P_o1, P_o2, P_o3, P_o4, P_o5, P_o6, P_o7,
         P_o8, P_o9, P_o10, P_o11, P_o12, P_o13, P_o14, P_o15]
    return [c[0:4], c[4:8], c[8:12], c[12:16]]


def _naive(A, B):
    return [[sum(A[i][k] * B[k][j] for k in range(4)) for j in range(4)] for i in range(4)]


if __name__ == "__main__":
    import random
    # all pairs of basis matrices: the 4096 coefficient identities
    for x in range(16):
        for y in range(16):
            A = [[Fraction(int(4 * i + j == x)) for j in range(4)] for i in range(4)]
            B = [[Fraction(int(4 * i + j == y)) for j in range(4)] for i in range(4)]
            C = multiply_4x4_202(A, B)
            assert all(type(v) is Fraction for row in C for v in row)
            assert C == _naive(A, B), (x, y)
    # random integer matrices
    rng = random.Random(2026)
    for _ in range(1000):
        A = [[Fraction(rng.randint(-99, 99)) for _ in range(4)] for _ in range(4)]
        B = [[Fraction(rng.randint(-99, 99)) for _ in range(4)] for _ in range(4)]
        C = multiply_4x4_202(A, B)
        assert all(type(v) is Fraction for row in C for v in row)
        assert C == _naive(A, B)
    print("OK: all 4096 identities hold exactly; 1000 random products match.")
