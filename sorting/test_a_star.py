import heapq

# Define the start state and goal state
start_state = [[4.2, 3.3, 2.2, 1.1], [4.1, 3.2, 2.1, 1.1], [6, 5, 4.3, 3.1], [], [], []]
goal_state = [
    [],
    [],
    [],
    [6, 5, 4.3, 4.2],
    [4.1, 3.3, 3.2, 3.1],
    [2.2, 2.1, 1.1, 1],
]
goal_state_map = {
    goal_state[i][j]: (i, j)
    for i in range(len(goal_state))
    for j in range(len(goal_state[i]))
}
print(goal_state_map)

# Define a function to calculate the cost of moving a card from one stack to another
def calculate_cost(from_stack, to_stack):
    return 1


def calculate_moves_to_target(state):
    for i, stack in enumerate(state):
        for j, card in enumerate(stack):
            return


# Define a function to calculate the estimated cost to reach the goal state from a given state
def calculate_heuristic(state):
    # Calculate the number of cards that are out of place
    num_out_of_place = 0
    for i in range(len(state)):
        for j in range(len(state[i])):
            if (
                j >= len(goal_state[i])
                or j >= len(state[i])
                or state[i][j] != goal_state[i][j]
            ):
                val = state[i][j]
                target = goal_state_map[val]
                moves = len(state[i]) - j
                num_out_of_place += moves
    return num_out_of_place


# Define a function to generate the next states from a given state
def generate_next_states(state):
    next_states = []
    min_value = None
    for i in range(len(state)):
        for j in range(len(state[i])):
            if (
                j >= len(goal_state[i])
                or j >= len(state[i])
                or state[i][j] != goal_state[i][j]
            ):
                loc = (i, j)
                val = state[i][j]
                target = goal_state_map[val]
                moves = len(state[i]) - j
                if min_value is None or min_value[0] > moves:
                    min_value = (moves, i, target[0])
    if min_value[1] == len(state[i]) - 1:
        next_state = [stack[:] for stack in state]
        top_card = next_state[min_value[1]].pop()
        next_state[min_value[2]].append(top_card)
        return [next_state]
    for i in range(len(state) // 2):
        if i not in min_value[1:]:
            next_state = [stack[:] for stack in state]
            top_card = next_state[min_value[1]].pop()
            next_state[i].append(top_card)
            next_states.append(next_state)
    return next_states


# Define a function to check if a state is the goal state
def is_goal_state(state):
    return state == goal_state


def main():
    # Initialize the priority queue
    pq = []
    heapq.heappush(pq, (0, start_state, []))

    # Initialize the set of visited states
    visited = dict()
    count = 0
    # Loop until the priority queue is empty
    while len(pq) > 0:
        count += 1
        # Pop the state with the lowest estimated cost
        cost, state, path = heapq.heappop(pq)

        tuple_state = tuple([tuple(stack) for stack in state])
        if tuple_state not in visited:
            visited[tuple_state] = True
            next_states = generate_next_states(state)
            # Loop through the next states and calculate the cost to reach each state
            for next_state in next_states:
                next_cost = cost + calculate_cost(state, next_state)
                next_path = path + [next_state]
                next_estimated_cost = next_cost + calculate_heuristic(next_state)

                # If the next state is the goal state, print the path and return
                if is_goal_state(next_state):
                    print(next_path)
                    return

                # Add the next state to the priority queue
                heapq.heappush(pq, (next_estimated_cost, next_state, next_path))

    # If the loop completes, print a message indicating that the goal state was not found
    print("Goal state not found.")


main()
