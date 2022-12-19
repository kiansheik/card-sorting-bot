import random
from math import ceil, sqrt, floor

MOVE_DURATION = 5
READ_DURATION = 1


def rectangle_dimensions(area):
    width = floor(sqrt(area))
    height = floor(sqrt(area))
    while width * height < area:
        if width * (height + 1) >= area:
            height += 1
            break
        width += 1
    return (width, height)


class Stack:
    max_size = 60

    def __init__(self, elems=[], idx=None):
        self.elements = elems if elems else list()
        self.idx = idx
        self.in_group_val = False

    # Must check `is_empty()` before calling `read_top_element()`
    def read_top_element(self):
        if self.is_empty():
            return None
        return self.elements[-1]

    def is_empty(self):
        return len(self.elements) == 0

    def is_full(self):
        return len(self.elements) >= self.max_size

    def pop(self):
        if self.is_empty():
            return None
        return self.elements.pop()

    # Must check `is_full()` before each `push()` because fullness is random
    def push(self, element):
        if self.is_full():
            raise Exception("Cannot push element to full stack")
        self.elements.append(element)

    def __str__(self):
        return str(self.elements)

    def __len__(self):
        return len(self.elements)

    def get_elements(self):
        return self.elements

    def in_group(self, ig=None):
        if not ig:
            return self.in_group_val
        self.in_group_val = ig


class StackGroup(Stack):
    def __init__(self, *stacks):
        self.stacks = stacks
        self.num_stacks = len(stacks)
        self.current_stack = 0
        self.in_group_val = False
        for stack in self.stacks:
            stack.in_group(True)

    def __len__(self):
        return sum([len(s) for s in self.stacks])

    def get_elements(self):
        elems = []
        for stack in self.stacks:
            elems += stack.get_elements()
        return elems

    def in_group(self, ig=None):
        if ig is None:
            return self.in_group_val
        self.in_group_val = ig

    def push(self, element):
        # Check if the current stack is full
        if self.stacks[self.current_stack].is_full():
            # If the current stack is the last stack, raise an exception
            if self.current_stack == self.num_stacks - 1:
                raise Exception("All stacks are full")
            # Otherwise, move to the next stack
            else:
                self.current_stack += 1
        # Push the element to the current stack
        self.stacks[self.current_stack].push(element)

    def pop(self):
        # Check if the current stack is empty
        if self.stacks[self.current_stack].is_empty():
            # If the current stack is the first stack, return None
            if self.current_stack == 0:
                return None
            # Otherwise, move to the previous stack
            else:
                self.current_stack -= 1
        # Pop the element from the current stack
        return self.stacks[self.current_stack].pop()

    def read_top_element(self):
        # Check if the current stack is empty
        if self.stacks[self.current_stack].is_empty():
            # If the current stack is the first stack, return None
            if self.current_stack == 0:
                return None
            # Otherwise, move to the previous stack
            else:
                self.current_stack -= 1
        # Return the top element of the current stack
        return self.stacks[self.current_stack].read_top_element()

    def is_empty(self):
        # Check if the first stack is empty
        return self.stacks[0].is_empty()

    def is_full(self):
        # Check if the last stack is full
        return self.stacks[-1].is_full()

    def __str__(self):
        # Initialize an empty list to store the string representation of each stack
        stack_strs = []
        # Iterate over the stacks in the StackGroup
        for stack in self.stacks:
            # Add the string representation of the stack to the list
            stack_strs.append(str(stack))
        # Join the list of stack strings with a newline character between each stack
        return "\n".join(stack_strs)


class Pile:
    def __init__(self, num_stacks=10):
        if type(num_stacks) == int:
            self.stacks = [Stack(idx=i) for i in range(num_stacks)]
        else:
            self.stacks = [Stack(elems=x, idx=i) for i, x in enumerate(num_stacks)]
        self.group_map = dict()
        self.reset_movement_counters()
        self.read_cache = {
            i: {"empty": None, "full": None, "top": None} for i in range(self.length())
        }

    def reset_movement_counters(self):
        self.total_moves = 0
        self.uncached_reads = 0
        self.cached_reads = 0

    def get_stack(self, stack_idx):
        return self.group_map.get(stack_idx, self.stacks[stack_idx])

    def put_stack(self, stack):
        self.stacks.append(stack)
        stack_idx = self.length() - 1
        self.put_read_cache(stack_idx, "empty", None)
        self.put_read_cache(stack_idx, "full", None)
        self.put_read_cache(stack_idx, "top", None)

    def put_stack_group(self, stack, stack_idx):
        self.group_map[stack_idx] = stack
        self.put_read_cache(stack_idx, "empty", None)
        self.put_read_cache(stack_idx, "full", None)
        self.put_read_cache(stack_idx, "top", None)

    def get_read_cache(self, stack_idx, name):
        stack = self.get_stack(stack_idx)
        if type(stack) == StackGroup:
            stack_idx = f"{stack_idx}_group"
        return self.read_cache[stack_idx][name]

    def put_read_cache(self, stack_idx, name, val):
        stack = self.get_stack(stack_idx)
        if type(stack) == StackGroup:
            stack_idx = f"{stack_idx}_group"
        if stack_idx not in self.read_cache.keys():
            self.read_cache[stack_idx] = {"empty": None, "full": None, "top": None}
        self.read_cache[stack_idx][name] = val

    def comparator(self, a, b):
        return a < b

    def length(self):
        return len(self.stacks)

    def size(self):
        return sum([len(stack) for stack in self.stacks])

    def read_top_element(self, stack_idx):
        top = self.get_read_cache(stack_idx, "top")
        if top is not None:
            self.cached_reads += 1
            return top
        top = self.get_stack(stack_idx).read_top_element()
        self.put_read_cache(stack_idx, "top", top)
        if top is not None:
            self.put_read_cache(stack_idx, "empty", False)
        self.uncached_reads += 1
        return top

    def is_full(self, stack_idx):
        full = self.get_read_cache(stack_idx, "full")
        if full is not None:
            self.cached_reads += 1
            return full
        full = self.get_stack(stack_idx).is_full()
        self.put_read_cache(stack_idx, "full", full)
        if full:
            self.put_read_cache(stack_idx, "empty", False)
        self.uncached_reads += 1
        return full

    def is_empty(self, stack_idx):
        empty = self.get_read_cache(stack_idx, "empty")
        if empty is not None:
            self.cached_reads += 1
            return empty
        empty = self.get_stack(stack_idx).is_empty()
        self.put_read_cache(stack_idx, "empty", empty)
        if empty:
            self.put_read_cache(stack_idx, "full", False)
        self.uncached_reads += 1
        return empty

    def move_element(self, src_stack, dest_stack):
        elem = self.get_stack(src_stack).pop()
        self.get_stack(dest_stack).push(elem)

        # Update read cache with values we can assume
        self.put_read_cache(src_stack, "full", False)
        self.put_read_cache(src_stack, "empty", None)
        self.put_read_cache(src_stack, "top", None)

        self.put_read_cache(dest_stack, "full", None)
        self.put_read_cache(dest_stack, "empty", False)
        self.put_read_cache(dest_stack, "top", elem)

        self.total_moves += 1

    def __str__(self):
        # Initialize an empty list to store the string representation of each stack
        stack_strs = []
        # Initialize a variable to store the maximum number of elements in any stack
        max_elements = 0
        # Iterate over the stacks in the pile
        for stack in self.stacks:
            # Find the maximum number of elements in any stack
            max_elements = max(max_elements, len(stack))
        # Iterate over the elements in each stack in reverse order
        for i in range(max_elements - 1, -1, -1):
            # Initialize an empty string to store the string representation of the elements in this row
            row_str = ""
            # Iterate over the stacks in the pile
            for stack in self.stacks:
                # If the stack has an element at this position, add it to the row string
                if i < len(stack):
                    row_str += f"{stack.get_elements()[i]:^7}"
                # If the stack does not have an element at this position, add the placeholder to the row string
                else:
                    row_str += "  ---  "
            # Add the row string to the list of stack strings
            stack_strs.append(row_str)
        # Join the list of stack strings with a newline character between each row
        return "\n".join(stack_strs)

    def sort(self):
        # Merge two sorted stacks into a single sorted stack
        def merge(stack1_idx, stack2_idx, swap_idx):
            num_elems = 0
            while not self.is_empty(stack1_idx) and not self.is_empty(stack2_idx):
                if self.comparator(
                    self.read_top_element(stack1_idx), self.read_top_element(stack2_idx)
                ):
                    self.move_element(stack1_idx, swap_idx)
                else:
                    self.move_element(stack2_idx, swap_idx)
                num_elems += 1
            # Move remaining elements from stack1 to merged_stack
            while not self.is_empty(stack1_idx):
                self.move_element(stack1_idx, swap_idx)
                num_elems += 1
            # Move remaining elements from stack2 to merged_stack
            while not self.is_empty(stack2_idx):
                self.move_element(stack2_idx, swap_idx)
                num_elems += 1
            # Create a StackGroup with stack1 and stack2, then replace stack1 inside of the pile with this group
            stack_group = StackGroup(
                self.get_stack(stack1_idx), self.get_stack(stack2_idx)
            )
            self.group_map[stack1_idx] = stack_group
            # Move elements from merged_stack back to stack1 and stack2 in reverse order
            for i in range(num_elems):
                self.move_element(swap_idx, stack1_idx)

        # Divide and merge the stacks until they are sorted
        def divide_and_merge(start_idx, end_idx, swap_idx):
            if start_idx == end_idx:
                return
            mid = (start_idx + end_idx) // 2
            divide_and_merge(start_idx, mid, swap_idx)
            divide_and_merge(mid + 1, end_idx, swap_idx)
            merge(start_idx, mid + 1, swap_idx)

        self.group_map = dict()
        self.reset_movement_counters()
        swap_size = ceil(self.size() / Stack.max_size)
        # We need enough empty stacks to store the total number of elements
        swap_stacks = [
            Stack() for _ in range(swap_size)
        ]  # Add a placehold and swap if it's an odd length of stacks, otherwise just add the swap
        p.put_stack(StackGroup(*swap_stacks))
        num_stacks = self.length()
        assert num_stacks >= 3
        swap_idx = num_stacks - 1
        last_stack_idx = num_stacks - 2
        # Sort each stack up to the swap
        for stack_idx in range(swap_idx):
            self.sort_stack(stack_idx, swap_idx)
        # Merge each sorted stack
        divide_and_merge(0, last_stack_idx, swap_idx)
        # Remove the empty stack from the end of the pile
        self.stacks.pop()
        self.group_map = dict()
        t_size = self.size()
        complexity = ceil(self.total_moves / t_size)
        total_stacks = swap_size + self.length()
        print(
            f"Sorted {t_size} items in {self.length()} stacks with {Stack.max_size-1} elements per stack and a swap of {swap_size} Stacks ({total_stacks} total) in {self.total_moves} moves O({complexity}n) ({self.uncached_reads}/{self.cached_reads}  Uncached/Cached reads)"
        )
        dim = rectangle_dimensions(total_stacks)
        print(
            f"This requires at least a {dim[0]}x{dim[1]} grid in size and estimated {(self.total_moves * MOVE_DURATION + (self.uncached_reads * READ_DURATION))/60/60/24} days to complete"
        )
        self.reset_movement_counters()

    def sort_stack(self, stack_idx, swap_idx):
        # Buffer only needs to hold one car, like a variable, so we can use any other stack in the pile
        # This ensures it will be a stack other than the two being used. This assumes that a stack always has at least
        # 1 less than the maximum card value

        # Find a buffer stack to use which is not full and not in a group
        # This stack can be any stack because it is used to store only 1 card at a time
        for i in range(swap_idx):
            if stack_idx != i:
                if not self.get_stack(i).in_group() and not self.is_full(i):
                    buffer_idx = i
                    break
        while not self.is_empty(stack_idx):
            if not self.is_empty(swap_idx) and self.comparator(
                self.read_top_element(stack_idx), self.read_top_element(swap_idx)
            ):
                self.move_element(stack_idx, buffer_idx)
                self.move_element(swap_idx, stack_idx)
                while not self.is_empty(swap_idx) and self.comparator(
                    self.read_top_element(buffer_idx), self.read_top_element(swap_idx)
                ):
                    self.move_element(swap_idx, stack_idx)
                self.move_element(buffer_idx, swap_idx)
            self.move_element(stack_idx, swap_idx)
        # Move elements from swap back to stack in reverse order
        while not self.is_empty(swap_idx):
            self.move_element(swap_idx, stack_idx)


p = Pile(
    [
        [ceil(random.random() * 30_000) for _ in range(ceil((Stack.max_size - 1)))]
        for _ in range(5 * 5)
    ]
)
print(p, "\n")
p.sort()
print(p)
