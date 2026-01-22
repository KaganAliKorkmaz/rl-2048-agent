import random


class Game2048Env:
    def __init__(self, size=4, max_spawn_tile=None):
        self.size = size
        self.max_spawn_tile = max_spawn_tile
        self.board = None
        self.score = 0
        self.done = False
        self.reset()

    def reset(self):
        self.board = [[0] * self.size for _ in range(self.size)]
        self.score = 0
        self.done = False
        self._add_random_tile()
        self._add_random_tile()
        return self.get_state()

    def get_state(self):
        return [row[:] for row in self.board]

    def step(self, action, corner="top_left", alpha=0.001, merge_scale=0.1, no_move_penalty=-0.1):
        """
        action: 0=up, 1=down, 2=left, 3=right
        Returns: next_state, reward, done, info
        """
        if self.done:
            raise RuntimeError("Game is over, call reset()")

        potential_before = self._corner_potential(corner=corner)

        if action == 0:
            changed, gain = self._move_up()
        elif action == 1:
            changed, gain = self._move_down()
        elif action == 2:
            changed, gain = self._move_left()
        elif action == 3:
            changed, gain = self._move_right()
        else:
            raise ValueError(f"Invalid action: {action}")

        base_reward = gain * merge_scale

        if changed:
            self._add_random_tile()
        else:
            base_reward += no_move_penalty

        potential_after = self._corner_potential(corner=corner)
        shaped_reward = base_reward + alpha * (potential_after - potential_before)

        if not self._can_move():
            self.done = True

        return self.get_state(), shaped_reward, self.done, {"score": self.score}

    def render(self):
        print("-" * (self.size * 6 + 1))
        for row in self.board:
            line = "|"
            for v in row:
                if v == 0:
                    line += "     |"
                else:
                    line += f"{v:5}|"
            print(line)
            print("-" * (self.size * 6 + 1))

    def _add_random_tile(self):
        empty = [
            (i, j)
            for i in range(self.size)
            for j in range(self.size)
            if self.board[i][j] == 0
        ]
        if not empty:
            return
        i, j = random.choice(empty)
        
        if self.max_spawn_tile is not None:
            max_value = self.max_spawn_tile
            spawn_value = 4 if random.random() < 0.1 else 2
            if spawn_value > max_value:
                spawn_value = 2 if max_value >= 2 else 2
            self.board[i][j] = spawn_value
        else:
            self.board[i][j] = 4 if random.random() < 0.1 else 2

    def _compress_and_merge_line(self, line):
        """
        Slide a single row/column to the left and merge.
        Returns: (new_line, score_gain)
        """
        tiles = [x for x in line if x != 0]
        new_tiles = []
        score_gain = 0
        skip = False

        for i in range(len(tiles)):
            if skip:
                skip = False
                continue
            if i + 1 < len(tiles) and tiles[i] == tiles[i + 1]:
                merged = 2 * tiles[i]
                new_tiles.append(merged)
                score_gain += merged
                skip = True
            else:
                new_tiles.append(tiles[i])

        new_tiles += [0] * (self.size - len(new_tiles))
        return new_tiles, score_gain

    def _move_left(self):
        changed = False
        total_gain = 0
        new_board = []
        for row in self.board:
            new_row, gain = self._compress_and_merge_line(row)
            if new_row != row:
                changed = True
            total_gain += gain
            new_board.append(new_row)
        self.board = new_board
        self.score += total_gain
        return changed, total_gain

    def _move_right(self):
        changed = False
        total_gain = 0
        new_board = []
        for row in self.board:
            rev = list(reversed(row))
            new_rev, gain = self._compress_and_merge_line(rev)
            new_row = list(reversed(new_rev))
            if new_row != row:
                changed = True
            total_gain += gain
            new_board.append(new_row)
        self.board = new_board
        self.score += total_gain
        return changed, total_gain

    def _move_up(self):
        changed = False
        total_gain = 0
        cols = [[self.board[i][j] for i in range(self.size)] for j in range(self.size)]
        new_cols = []
        for col in cols:
            new_col, gain = self._compress_and_merge_line(col)
            if new_col != col:
                changed = True
            total_gain += gain
            new_cols.append(new_col)
        for j in range(self.size):
            for i in range(self.size):
                self.board[i][j] = new_cols[j][i]
        self.score += total_gain
        return changed, total_gain

    def _move_down(self):
        changed = False
        total_gain = 0
        cols = [[self.board[i][j] for i in range(self.size)] for j in range(self.size)]
        new_cols = []
        for col in cols:
            rev = list(reversed(col))
            new_rev, gain = self._compress_and_merge_line(rev)
            new_col = list(reversed(new_rev))
            if new_col != col:
                changed = True
            total_gain += gain
            new_cols.append(new_col)
        for j in range(self.size):
            for i in range(self.size):
                self.board[i][j] = new_cols[j][i]
        self.score += total_gain
        return changed, total_gain

    def _simulate_left(self, board):
        total_gain = 0
        new_board = []
        for row in board:
            new_row, gain = self._compress_and_merge_line(row)
            new_board.append(new_row)
            total_gain += gain
        changed = new_board != board
        return new_board, total_gain, changed

    def _simulate_right(self, board):
        total_gain = 0
        new_board = []
        for row in board:
            rev = list(reversed(row))
            new_rev, gain = self._compress_and_merge_line(rev)
            new_row = list(reversed(new_rev))
            new_board.append(new_row)
            total_gain += gain
        changed = new_board != board
        return new_board, total_gain, changed

    def _simulate_up(self, board):
        total_gain = 0
        size = self.size
        cols = [[board[i][j] for i in range(size)] for j in range(size)]
        new_cols = []
        for col in cols:
            new_col, gain = self._compress_and_merge_line(col)
            new_cols.append(new_col)
            total_gain += gain
        new_board = [[0] * size for _ in range(size)]
        for j in range(size):
            for i in range(size):
                new_board[i][j] = new_cols[j][i]
        changed = new_board != board
        return new_board, total_gain, changed

    def _simulate_down(self, board):
        total_gain = 0
        size = self.size
        cols = [[board[i][j] for i in range(size)] for j in range(size)]
        new_cols = []
        for col in cols:
            rev = list(reversed(col))
            new_rev, gain = self._compress_and_merge_line(rev)
            new_col = list(reversed(new_rev))
            new_cols.append(new_col)
            total_gain += gain
        new_board = [[0] * size for _ in range(size)]
        for j in range(size):
            for i in range(size):
                new_board[i][j] = new_cols[j][i]
        changed = new_board != board
        return new_board, total_gain, changed

    def _simulate_move(self, board, action):
        if action == 0:
            return self._simulate_up(board)
        elif action == 1:
            return self._simulate_down(board)
        elif action == 2:
            return self._simulate_left(board)
        elif action == 3:
            return self._simulate_right(board)
        else:
            raise ValueError(f"Invalid action: {action}")

    def _can_move(self):
        for action in range(4):
            board_copy = [row[:] for row in self.board]
            _, _, changed = self._simulate_move(board_copy, action)
            if changed:
                return True
        return False
    def _build_corner_weights(self, corner="top_left"):
        base = [
            [16,  8,  4,  2],
            [ 8,  4,  2,  1],
            [ 4,  2,  1, 0.5],
            [ 2,  1, 0.5, 0.25],
        ]
        if corner == "top_left":
            return base
        elif corner == "top_right":
            return [list(reversed(row)) for row in base]
        elif corner == "bottom_left":
            return list(reversed(base))
        elif corner == "bottom_right":
            return [list(reversed(row)) for row in reversed(base)]
        else:
            raise ValueError("Invalid corner")

    def _corner_potential(self, corner="top_left"):
        weights = self._build_corner_weights(corner)
        val = 0.0
        for i in range(self.size):
            for j in range(self.size):
                val += self.board[i][j] * weights[i][j]
        return val
