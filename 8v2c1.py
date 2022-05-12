import numpy as np
import heapq
import time
import matplotlib.pyplot as plt

mapsDifficulty = ['trivial', 'veryEasy', 'easy', 'doable', 'oh boy', 'Impossible']
maps = [
    [[1, 2, 3],
     [4, 5, 6],
     [7, 8, 0]],

    [[1, 2, 3],
     [4, 5, 6],
     [7, 0, 8]],

    [[1, 2, 0],
     [4, 5, 3],
     [7, 8, 6]],

    [[0, 1, 2],
     [4, 5, 3],
     [7, 8, 6]],

    [[8, 7, 1],
     [6, 0, 2],
     [5, 4, 3]],

    [[8, 7, 1],
     [6, 0, 2],
     [5, 4, 3]]
]

eight_goal_state = [[1, 2, 3], [4, 5, 6], [7, 8, 0]]


class PriorityQueue:
    """
    Priority queue for A-star search"""

    def __init__(self, start, cost):
        self.states = {}
        self.q = []
        self.add(start, cost)

    def add(self, state, cost):
        """ push the new state and cost to get there onto the heap"""
        heapq.heappush(self.q, (cost, state))
        self.states[state] = cost

    def pop(self):
        (cost, state) = heapq.heappop(self.q)  # get cost of getting to explored state
        self.states.pop(state)  # and remove from frontier
        return (cost, state)

    def replace(self, state, cost):
        """ found a cheaper route to `state`, replacing old cost with new `cost` """
        self.states[state] = cost
        for i, (oldcost, oldstate) in enumerate(self.q):
            if oldstate == state and oldcost > cost:
                self.q[i] = (cost, state)
                heapq._siftdown(self.q, 0, i)  # now i is posisbly out of order; restore
        return


def heuristic_manhattan(state, goal):
    """calculates the manhattan distance from the current state to the goal"""
    # Your code here.
    manhattan_dist = 0
    x, y = state.shape
    for i in range(x):
        for j in range(y):
            idx = np.argwhere(goal == state[i][j])
            manhattan_dist += abs(i - idx[0][0]) + abs(j - idx[0][1])
    return manhattan_dist


def heuristic_misplaced(state, goal):
    """calculates the number of misplaced tiles from current state to the goal"""
    # Your code here.
    misplaced_dist = 0
    x, y = state.shape
    for i in range(x):
        for j in range(y):
            if state[i][j] != goal[i][j]:
                misplaced_dist += 1
    return misplaced_dist


def heuristic_ucs(state, goal):
    return 0


def check_parity(state1, state2):
    """checks to see if the parity of the two states is the same
    the 8-tile problem will not be solvable if they are not"""
    x, y = state1.shape

    pair1, pair2 = 0, 0
    state1_copy = np.reshape(state1, x * y)
    state2_copy = np.reshape(state2, x * y)
    state1_copy = np.delete(state1_copy, list(state1_copy).index(0))
    state2_copy = np.delete(state2_copy, list(state2_copy).index(0))
    for i in range(x * y - 2):
        for j in range(i + 1, x * y - 1):
            if state1_copy[i] > state1_copy[j]:
                pair1 += 1
            if state2_copy[i] > state2_copy[j]:
                pair2 += 1

    return pair1 % 2 == pair2 % 2


def adjacent_states(state):
    """what are all the successors of this state? depends on location of the 0 (blank tile)"""
    adjacent = []
    loc0 = [int(np.where(state == 0)[i]) for i in range(2)]
    if loc0[0] > 0:
        # If row of 0 is > 0, then we can move 0 up
        swap = np.copy(state)
        newloc = [loc0[0] - 1, loc0[1]]
        swap[loc0[0], loc0[1]] = state[newloc[0], newloc[1]]
        swap[newloc[0], newloc[1]] = 0
        adjacent.append(swap)
    if loc0[0] < (state.shape[0] - 1):
        # If row of 0 is not bottom, then can move 0 down
        swap = np.copy(state)
        newloc = [loc0[0] + 1, loc0[1]]
        swap[loc0[0], loc0[1]] = state[newloc[0], newloc[1]]
        swap[newloc[0], newloc[1]] = 0
        adjacent.append(swap)
    if loc0[1] > 0:
        # If column of 0 is > 0, then we can move 0 left
        swap = np.copy(state)
        newloc = [loc0[0], loc0[1] - 1]
        swap[loc0[0], loc0[1]] = state[newloc[0], newloc[1]]
        swap[newloc[0], newloc[1]] = 0
        adjacent.append(swap)
    if loc0[1] < (state.shape[1] - 1):
        # If column of 0 is not far-right, then we can move 0 right
        swap = np.copy(state)
        newloc = [loc0[0], loc0[1] + 1]
        swap[loc0[0], loc0[1]] = state[newloc[0], newloc[1]]
        swap[newloc[0], newloc[1]] = 0
        adjacent.append(swap)
    return adjacent


def astar_search(start, goal, heuristic):
    sta = time.time()

    if not check_parity(start, goal):
        return {}, 0, 0, 0

    temp = np.copy(start)
    x, y = start.shape

    frontier = PriorityQueue(tuple(temp.reshape(x * y)), heuristic(start, goal))
    previous = {tuple(temp.reshape(x * y)): None}
    explored = {}
    expand = 0
    while frontier:
        s = frontier.pop()
        s_array = np.array(list(s[1])).reshape(x, y)
        expand += 1
        if (s_array == goal).all():
            end = time.time()
            return previous, s[0], expand, end - sta
        explored[s[1]] = s[0]
        for s2 in adjacent_states(s_array):
            temp = np.copy(s2)
            s2_tuple = tuple(temp.reshape(x * y))
            newcost = explored[s[1]] - heuristic(s_array, goal) + 1 + heuristic(s2, goal)
            if (s2_tuple not in explored) and (s2_tuple not in frontier.states):
                frontier.add(s2_tuple, newcost)
                previous[s2_tuple] = s[1]
            elif (s2_tuple in frontier.states) and (frontier.states[s2_tuple] > newcost):
                frontier.replace(s2_tuple, newcost)
                previous[s2_tuple] = s[1]


def print_moves(moves, goal, puzzle_len, mode):
    tmp_goal = goal.copy()
    goal = tuple(goal.reshape(puzzle_len * puzzle_len))
    res = []
    while goal:
        res.append(goal)
        goal = moves[goal]
    res.reverse()
    cnt = 0
    for sta in res:
        tmp = np.array(sta).reshape(puzzle_len, puzzle_len)
        if mode == 1:
            hn = heuristic_ucs(tmp, tmp_goal)
        elif mode == 2:
            hn = heuristic_misplaced(tmp, tmp_goal)
        else:
            hn = heuristic_manhattan(tmp, tmp_goal)
        print(f"Current steps: {cnt}, Heuristic Function Value: {hn}")
        print(tmp)
        print()
        cnt += 1


def init_default_puzzle_mode():
    selected_difficulty = input(
        "You wish to use a default puzzle. Please enter a desired difficulty on a scale from 0 to 5." + '\n')
    if '0' <= selected_difficulty <= '5':
        difficulty = int(selected_difficulty)
        print(f"Difficulty of '{mapsDifficulty[difficulty]}' selected.")
        return maps[difficulty]


def generate_puzzle(goal, n_moves=4):
    """generate an 8-tile puzzle solve-able in n_moves from goal"""
    puzzle = np.copy(goal)
    previous = [goal]
    for k in range(n_moves):
        adjacent = adjacent_states(puzzle)
        ind_pop = []
        for a in range(len(adjacent)):
            if any((previous_state == adjacent[a]).all() for previous_state in previous):
                ind_pop.append(a)
        if ind_pop:
            ind_pop.reverse()
            [adjacent.pop(i) for i in ind_pop]
        puzzle = adjacent[np.random.randint(len(adjacent))]
        previous.append(puzzle)
    return puzzle, previous


def moves_steps(begin, end, goal):
    move, cost_manhattan, nexp_manhattan = [], [], []
    ucs, manhattan, misplaced = [], [], []
    ucs_t, manhattan_t, misplaced_t = [], [], []
    for i in range(begin, end):
        puzzle, previous = generate_puzzle(goal, n_moves=i)
        n_moves, cost, nexp, time_cost = astar_search(puzzle, goal, heuristic_ucs)

        ucs.append((cost, nexp, time_cost))

        n_moves, cost, nexp, time_cost = astar_search(puzzle, goal, heuristic_manhattan)
        move.append(i)

        manhattan.append((cost, nexp, time_cost))
        n_moves, cost, nexp, time_cost = astar_search(puzzle, goal, heuristic_misplaced)

        misplaced.append((cost, nexp, time_cost))
    ucs.sort(key=lambda x: x[0])
    manhattan.sort(key=lambda x: x[0])
    misplaced.sort(key=lambda x: x[0])
    n_moves, cost_manhattan, nexp_manhattan = [], [], []
    cost_misplaced, nexp_misplaced = [], []
    cost_ucs, nexp_ucs = [], []
    for tp in ucs:
        cost, nexp, t = tp
        cost_ucs.append(cost)
        nexp_ucs.append(nexp)
        ucs_t.append(t)
    for tp in manhattan:
        cost, nexp, t = tp
        cost_manhattan.append(cost)
        nexp_manhattan.append(nexp)
        manhattan_t.append(t)
    for tp in misplaced:
        cost, nexp, t = tp
        cost_misplaced.append(cost)
        nexp_misplaced.append(nexp)
        misplaced_t.append(t)

    return move, cost_ucs, nexp_ucs, cost_manhattan, nexp_manhattan, cost_misplaced, nexp_misplaced, ucs_t, manhattan_t, misplaced_t


# ------------------------------------------------- Report Plots generations (starts) ------------------------------------------------- 

def plot_different_heuristic_function(cost_ucs, nexp_ucs, cost_manhattan, nexp_manhattan, cost_misplaced,
                                      nexp_misplaced):
    print(n_moves, nexp_ucs, nexp_manhattan, nexp_misplaced)
    plt.figure()
    plt.xlabel('n_moves')
    plt.ylabel('node expands')
    l1, = plt.plot(cost_ucs, nexp_ucs)
    l2, = plt.plot(cost_manhattan, nexp_manhattan)
    l3, = plt.plot(cost_misplaced, nexp_misplaced)

    plt.legend(handles=[l1, l2, l3], labels=['ucs', 'manhattan', 'misplaced'], loc='best')
    plt.savefig("res.png")
    plt.figure(num=3, figsize=(12, 8))

    plt.show()


def plot_different_heuristic_function_time(cost_ucs, nexp_ucs, cost_manhattan, nexp_manhattan, cost_misplaced,
                                           nexp_misplaced):
    print(n_moves, nexp_ucs, nexp_manhattan, nexp_misplaced)
    plt.figure()
    plt.xlabel('n_moves')
    plt.ylabel('time cost')
    l1, = plt.plot(cost_ucs, nexp_ucs)
    l2, = plt.plot(cost_manhattan, nexp_manhattan)
    l3, = plt.plot(cost_misplaced, nexp_misplaced)

    plt.legend(handles=[l1, l2, l3], labels=['ucs', 'manhattan', 'misplaced'], loc='best')
    plt.savefig("res2.png")
    plt.figure(num=3, figsize=(12, 8))

    plt.show()

# ------------------------------------------------- Report Plots generations (ends) ------------------------------------------------- 

def main():
    #flag = 1
    #while flag:
            puzzle_mode = input("Welcome to an 8-Puzzle Solver. Type '1' to use a default puzzle, or '2' to create your own."
                                + '\n')
            if puzzle_mode == "1":
                user_puzzle = (init_default_puzzle_mode())
                target = eight_goal_state
                puzzle_len = 3
            if puzzle_mode == "2":
                print("Enter your puzzle, using a zero to represent the blank. " +
                      "Please only enter valid 8-puzzles. Enter the puzzle demilimiting " +
                      "the numbers with a space. RET only when finished." + '\n')
                puzzle_row_one = input("Enter the 1th row: ")
                puzzle_row_one = puzzle_row_one.split()
                puzzle_len = len(puzzle_row_one)
                user_puzzle = [[int(i) for i in puzzle_row_one]]
                for i in range(1, puzzle_len):
                    puzzle_row = input(f"Enter the {i}th row: ")
                    puzzle_row = puzzle_row.split()
                    if len(puzzle_row) != puzzle_len:
                        exit(0)
                    user_puzzle.append([int(i) for i in puzzle_row])
                target = [(i + 1) % (puzzle_len * puzzle_len) for i in range(puzzle_len * puzzle_len)]

            h = input("Select algorithm.\n (1) for Uniform Cost Search,\n (2) for the Misplaced Tile Heuristic, or\n"
                      " (3) the Manhattan Distance Heuristic.\n")
            if h == '1':
                heuristic = heuristic_ucs
            elif h == '2':
                heuristic = heuristic_misplaced
            else:
                heuristic = heuristic_manhattan
            user_puzzle = np.array(user_puzzle)
            target = np.array(target)
            target = target.reshape(puzzle_len, puzzle_len)
            print(user_puzzle)
            print(target)

            moves, steps, nexp, _ = astar_search(user_puzzle, target, heuristic)

            print_moves(moves, target, puzzle_len, h)
            print(steps)
#            flag = int(input("Enter 1 to repeat, 0 to end: > "))


if __name__ == "__main__":
    #Generate plots for report
    n_moves, cost_ucs, nexp_ucs, cost_manhattan, nexp_manhattan, cost_misplaced, nexp_misplaced, ucs_t, manhattan_t, misplaced_t = moves_steps(
        2, 26, eight_goal_state)
    plot_different_heuristic_function(cost_ucs, nexp_ucs, cost_manhattan, nexp_manhattan, cost_misplaced,
                                      nexp_misplaced)
    plot_different_heuristic_function_time(cost_ucs, ucs_t, cost_manhattan, manhattan_t, cost_misplaced, misplaced_t)

    main()
