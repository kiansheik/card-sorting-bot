import heapq

# Define the start state and goal state
start_state = [[4, 3, 2, 1], [4, 3, 2, 1], [6, 5, 4, 3], [], [], []]
goal_state = [
    [],
    [],
    [],
    [6, 5, 4, 4],
    [4, 3, 3, 3],
    [2, 2, 1, 1],
]


# Define a function to calculate the cost of moving a card from one stack to another
def calculate_cost(from_stack, to_stack):
    return 1


# Define a function to calculate the estimated cost to reach the goal state from a given state
def calculate_heuristic(state):
    # Calculate the number of cards that are out of place
    num_out_of_place = sum(
        [
            1
            for i in range(len(state))
            for j in range(len(state[i]))
            if state[i][j] != goal_state[i][j]
        ]
    )
    return num_out_of_place


# Define a function to generate the next states from a given state
def generate_next_states(state):
    next_states = []
    for i in range(len(state)):
        for j in range(len(state)):
            if i != j and len(state[i]) > 0:
                next_state = [stack[:] for stack in state]
                top_card = next_state[i].pop(0)
                next_state[j].insert(0, top_card)
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
    visited = set()

    # Loop until the priority queue is empty
    while len(pq) > 0:
        # Pop the state with the lowest estimated cost
        cost, state, path = heapq.heappop(pq)

        # If the state has not been visited, add it to the visited set and generate the next states
        if state not in visited:
            visited.add(state)
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
