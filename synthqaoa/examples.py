import nchoosek


def exact_cover():
    ########################################
    # Solve the following exact cover      #
    # problem with NchooseK:               #
    #                                      #
    # Given a set E = {a,b,c,d,e,f,g}      #
    # and the following subsets of E       #
    #                                      #
    #   S1 = {b,c,e,f}                     #
    #   S2 = {a,d,e}                       #
    #   S3 = {a,d,e,g}                     #
    #   S4 = {a,g,f}                       #
    #   S5 = {c,f}                         #
    #   S6 = {b,g}                         #
    #                                      #
    # find a subset of {S1,S2,S3,S4,S5,S6} #
    # such that each element of E appears  #
    # exactly once.                        #
    #                                      #
    # By Scott Pakin <pakin@lanl.gov>      #
    ########################################
    # S2,5,6 covers

    env = nchoosek.Environment()
    S = [None] + [env.register_port('S%d' % i)
                for i in range(1, 7)]  # Number from 1.
    env.nck([S[2], S[3], S[4]], {1})
    env.nck([S[1], S[6]], {1})
    env.nck([S[1], S[5]], {1})
    env.nck([S[2], S[3]], {1})
    env.nck([S[1], S[2], S[3]], {1})
    env.nck([S[1], S[4], S[5]], {1})
    env.nck([S[3], S[4], S[6]], {1})
    result = env.solve()
    soln = result.solutions[0]
    print('Exact vertex cover: %s' %
        (' '.join(sorted([k for k, v in soln.items() if v]))))

    return env

def min_vert_cover():
    ######################################
    # Solve the following minimum vertex #
    # cover problem with NchooseK:       #
    #                                    #
    #   6 - 4 - 5 - 1                    #
    #       |   |  /                     #
    #       3 -  2                       #
    #                                    #
    # By Scott Pakin <pakin@lanl.gov>    #
    ######################################

    env = nchoosek.Environment()
    verts = [env.register_port(str(i + 1)) for i in range(6)]
    for u, v in [(1, 2),
                (1, 5),
                (2, 3),
                (2, 5),
                (3, 4),
                (4, 6)]:
        env.nck([verts[u - 1], verts[v - 1]], {1, 2})
    env.minimize(verts)
    result = env.solve()
    soln = result.solutions[0]
    print('Minimum vertex cover: %s' %
        ' '.join(sorted([v for v, b in soln.items() if b], key=int)))
    return env

def maxcut5():
    #######################################
    # Solve the following max-cut problem #
    # with NchooseK:                      #
    #                                     #
    #       A                             #
    #      / \                            #
    #     B - C                           #
    #     |   |                           #
    #     D - E                           #
    #                                     #
    # By Scott Pakin <pakin@lanl.gov>     #
    #######################################

    env = nchoosek.Environment()
    a = env.register_port('A')
    b = env.register_port('B')
    c = env.register_port('C')
    d = env.register_port('D')
    e = env.register_port('E')
    for edges in [(a, b),
                (a, c),
                (b, c),
                (b, d),
                (c, e),
                (d, e)]:
        env.different(edges[0], edges[1], soft=True)
    result = env.solve()
    soln = result.solutions[0]
    print('Partition 1: %s' %
        ' '.join(sorted([k for k, v in soln.items() if v])))
    print('Partition 2: %s' %
        ' '.join(sorted([k for k, v in soln.items() if not v])))
    return env

def maxcut4():
    #######################################
    # Solve the following max-cut problem #
    # with NchooseK:                      #
    #                                     #
    #     A - B                           #
    #     |   |                           #
    #     D - C                           #
    #                                     #
    #######################################

    env = nchoosek.Environment()
    a = env.register_port('A')
    b = env.register_port('B')
    c = env.register_port('C')
    d = env.register_port('D')
    for edges in [(a, b),
                (b, c),
                (c, d),
                (d, a)]:
        env.different(edges[0], edges[1], soft=True)

    result = env.solve()
    soln = result.solutions[0]
    print('Partition 1: %s' %
        ' '.join(sorted([k for k, v in soln.items() if v])))
    print('Partition 2: %s' %
        ' '.join(sorted([k for k, v in soln.items() if not v])))
    return env



ALL = {
    'exact_cover': exact_cover(),
    'min_vert_cover': min_vert_cover(),
    'max_cut_5': maxcut5(),
    'max_cut_4': maxcut4()}

EXACT_COVER = ALL['exact_cover']
MIN_VERT_COVER = ALL['min_vert_cover']
MAXCUT5 = ALL['max_cut_5']
MAXCUT4 = ALL['max_cut_4']