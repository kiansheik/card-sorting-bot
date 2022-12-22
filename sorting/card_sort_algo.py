import random
from math import ceil, sqrt, floor
from functools import cmp_to_key

MOVE_DURATION = 5
READ_DURATION = 2


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
            raise ValueError("Stack empty!")
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

    def size(self):
        return len(self.stacks)

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
            i: {"empty": None, "full": None, "top": [], "been_emptied": False}
            for i in range(self.length())
        }

    def reset_movement_counters(self):
        self.total_moves = 0
        self.uncached_reads = 0
        self.cached_reads = 0

    def get_stack(self, stack_idx, idx=None):
        if ".stack[" in str(stack_idx):
            vals = str(stack_idx).split(".stack[")
            stack_idx = vals[0]
            idx = int(vals[1].split("]")[0])
        stack = self.group_map.get(str(stack_idx), self.stacks[int(stack_idx)])
        if idx is None:
            return stack
        # Case when we want to directly access a Stack Group's element stack
        return stack.stacks[idx]

    def put_stack(self, stack):
        self.stacks.append(stack)
        stack_idx = str(self.length() - 1)
        self.read_cache[str(stack_idx)] = {
            "empty": None,
            "full": None,
            "top": [],
            "been_emptied": False,
        }

    def put_stack_group(self, stack, stack_idx):
        stack_idx = str(stack_idx)
        self.group_map[stack_idx] = stack
        self.read_cache[stack_idx] = {
            "empty": None,
            "full": None,
            "top": [],
            "been_emptied": False,
        }

    def swap(self):
        return self.stacks[-1]

    def swap_idx(self):
        return self.length() - 1

    def get_read_cache(self, stack_idx, name):
        stack = self.get_stack(stack_idx)
        if type(stack) == StackGroup:
            stack_idx = f"{stack_idx}_group"
        rc = self.read_cache.get(
            stack_idx, {"empty": None, "full": None, "top": [], "been_emptied": False}
        )["top" if name == "elements" else name]
        if name == "top":
            return rc[-1] if len(rc) > 0 else None
        return rc

    def put_read_cache(self, stack_idx, name, val):
        stack = self.get_stack(stack_idx)
        if type(stack) == StackGroup:
            stack_idx = f"{stack_idx}_group"
        stack_idx = str(stack_idx)
        if stack_idx not in self.read_cache.keys():
            self.read_cache[stack_idx] = {
                "empty": None,
                "full": None,
                "top": [],
                "been_emptied": False,
            }
        if name == "top":
            if val is None:
                if len(self.read_cache[stack_idx][name]) > 0:
                    self.read_cache[stack_idx][name].pop()
            else:
                self.read_cache[stack_idx][name].append(val)
        else:
            self.read_cache[stack_idx][name] = val

    def comparator(self, a, b):
        return a < b

    def key_cmp(self, x):
        def compare(a, b):
            if self.comparator(a, b):
                return -1
            elif self.comparator(b, a):
                return 1
            else:
                return 0

        return cmp_to_key(compare)(x)

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
        # If stack has been empty once, we know every element inside of it
        been_emptied = self.get_read_cache(stack_idx, "been_emptied")
        if been_emptied:
            full = len(self.get_read_cache(stack_idx, "elements")) >= Stack.max_size
        else:
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
        # If stack has been empty once, we know every element inside of it
        been_emptied = self.get_read_cache(stack_idx, "been_emptied")
        if been_emptied:
            empty = len(self.get_read_cache(stack_idx, "elements")) == 0
        else:
            empty = self.get_read_cache(stack_idx, "empty")
        if empty is not None:
            self.cached_reads += 1
            return empty
        empty = self.get_stack(stack_idx).is_empty()
        self.put_read_cache(stack_idx, "empty", empty)
        if empty:
            self.put_read_cache(stack_idx, "full", False)
            self.put_read_cache(stack_idx, "been_emptied", True)
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
        def merge(stack1_idx, stack2_idx, swap_idx, pile=self):
            num_elems = 0
            while not pile.is_empty(stack1_idx) and not pile.is_empty(stack2_idx):
                if pile.comparator(
                    pile.read_top_element(stack1_idx), pile.read_top_element(stack2_idx)
                ):
                    pile.move_element(stack1_idx, swap_idx)
                else:
                    pile.move_element(stack2_idx, swap_idx)
                num_elems += 1
            # Move remaining elements from stack1 to merged_stack
            while not pile.is_empty(stack1_idx):
                pile.move_element(stack1_idx, swap_idx)
                num_elems += 1
            # Move remaining elements from stack2 to merged_stack
            while not pile.is_empty(stack2_idx):
                pile.move_element(stack2_idx, swap_idx)
                num_elems += 1
            # Create a StackGroup with stack1 and stack2, then replace stack1 inside of the pile with this group
            stack_group = StackGroup(
                pile.get_stack(stack1_idx), pile.get_stack(stack2_idx)
            )
            pile.group_map[str(stack1_idx)] = stack_group
            # Move elements from merged_stack back to stack1 in reverse order
            for i in range(num_elems):
                pile.move_element(swap_idx, stack1_idx)

        # Divide and merge the stacks until they are sorted
        def divide_and_merge(start_idx, end_idx, swap_idx, pile=self):
            if start_idx == end_idx:
                return
            mid = (start_idx + end_idx) // 2
            divide_and_merge(start_idx, mid, swap_idx, pile=pile)
            divide_and_merge(mid + 1, end_idx, swap_idx, pile=pile)
            merge(start_idx, mid + 1, swap_idx, pile=pile)

        def sort_stack(stack_idx):
            # Buffer only needs to hold one car, like a variable, so we can use any other stack in the pile
            # This ensures it will be a stack other than the two being used. This assumes that a stack always has at least
            # 1 less than the maximum card value

            # Find a buffer stack to use which is not full and not in a group
            # This stack can be any stack because it is used to store only 1 card at a time
            swap_idx = self.swap_idx()
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
                        self.read_top_element(buffer_idx),
                        self.read_top_element(swap_idx),
                    ):
                        self.move_element(swap_idx, stack_idx)
                    self.move_element(buffer_idx, swap_idx)
                self.move_element(stack_idx, swap_idx)
            # Move elements from swap back to stack in reverse order
            while not self.is_empty(swap_idx):
                self.move_element(swap_idx, stack_idx)

        def sort_stack_immproved(src_idx):
            # We want to sort `stack` using the list of `swap_stacks` as space to sort
            swap_idx = self.swap_idx()
            swap_stacks = self.swap().stacks

            # The state will help us keep track of where we put each element from `stack`
            state = [list() for _ in swap_stacks]
            while not self.is_empty(src_idx):
                elem = self.read_top_element(src_idx)
                # Leave last stack for unsortable elements
                for i in range(len(state) - 1):
                    if len(state[i]) == 0 or not self.comparator(elem, state[i][-1]):
                        # Keep track internally
                        state[i].append(elem)
                        # Move element on the robot from one stack to another
                        self.move_element(src_idx, f"{swap_idx}.stack[{i}]")
                        # Let rest of loop know that we have placed elem and break
                        elem = None
                        break
                if elem is not None:
                    state[len(state) - 1].append(elem)
                    self.move_element(src_idx, f"{swap_idx}.stack[{len(state) - 1}]")
            tl = Pile(state)
            print(tl)
            while sum([len(stack) for stack in state[:-1]]) > 0:
                min_val = None
                for i, stack in enumerate(state[:-1]):
                    if len(stack) > 0 and (
                        min_val is None or not self.comparator(stack[-1], min_val[1])
                    ):
                        min_val = (i, stack.pop())
                self.move_element(f"{swap_idx}.stack[{min_val[0]}]", src_idx)
            print(self.get_stack(src_idx))
            print(tl)
            breakpoint()

        def sort_stack_quicksort(
            src_idx,
            swap_stack_idxs=None,
            src_count=float("Inf"),
            buffer_idx=None,
            final_idx=None,
            tl="start",
        ):
            # We want to sort `stack` using the list of `swap_stacks` as space to sort
            if tl == "start":
                swap_stack_idxs = [
                    f"{self.swap_idx()}.stack[{i}]" for i in range(self.swap().size())
                ]
                final_idx = f"{src_idx}"
                buffer_idx = f"{swap_stack_idxs[2]}"
            else:
                buffer_idx = f"{src_idx}"
            left_idx = swap_stack_idxs[0]
            right_idx = swap_stack_idxs[1]
            print(tl, src_idx, swap_stack_idxs, src_count, buffer_idx, final_idx)
            print("\tsrc:\t", self.get_stack(src_idx))
            print("")
            print("\tleft:\t", self.get_stack(left_idx))
            print("\tright:\t", self.get_stack(right_idx))
            print("\n\n")
            if src_count == 2:
                elems = self.get_read_cache(src_idx, "elements")
                if not self.comparator(elems[0], elems[1]):
                    self.move_element(src_idx, right_idx)
                    self.move_element(src_idx, final_idx)
                    self.move_element(right_idx, final_idx)
                else:
                    self.move_element(src_idx, final_idx)
                    self.move_element(src_idx, final_idx)
                return
            elif src_count == 1:
                self.move_element(src_idx, final_idx)
                return
            elif src_count < 1:
                return
            elems = self.get_read_cache(src_idx, "elements")
            pivot = None
            if len(elems) > 0:
                if src_count != float("Inf"):
                    elems = elems[-1 * src_count :]
                pivot = sorted(elems, key=self.key_cmp)[len(elems) // 2]
            left_count = 0
            right_count = 0
            i = 0
            while i < src_count and not self.is_empty(src_idx):
                elem = self.read_top_element(src_idx)
                if pivot is None:
                    pivot = elem
                if self.comparator(elem, pivot):
                    self.move_element(src_idx, left_idx)
                    left_count += 1
                else:
                    self.move_element(src_idx, right_idx)
                    right_count += 1
                i += 1
            print("\tsrc:\t", self.get_stack(src_idx))
            print("\tpivot:\t", pivot)
            print("\tleft:\t", self.get_stack(left_idx))
            print("\tright:\t", self.get_stack(right_idx))
            print("\n\n")
            breakpoint()
            sort_stack_quicksort(
                right_idx,
                swap_stack_idxs=[left_idx, buffer_idx],
                src_count=right_count,
                final_idx=final_idx,
                tl="right",
            )
            # while not self.is_empty(right_idx):
            #     self.move_element(right_idx, final_idx)
            sort_stack_quicksort(
                left_idx,
                swap_stack_idxs=[right_idx, buffer_idx],
                src_count=left_count,
                final_idx=final_idx,
                tl="left",
            )
            # for _ in range(left_count):
            #     self.move_element(left_idx, final_idx)

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
            sort_stack_quicksort(stack_idx)
            print(f"{stack_idx} sorted...")
            breakpoint()
        print(self.total_moves, self.uncached_reads)
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


p = Pile(
    [
        [ceil(random.random() * 30_000) for _ in range(ceil((Stack.max_size - 1)))]
        for _ in range(4 * 2)
    ]
)
print(p, "\n")
p.sort()
print(p)
