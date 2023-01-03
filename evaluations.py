def eval1(board):
    """
    Evaluation function that weights chips by height

    P0 is max player
    P1 is min player
    """
    ROWS = len(board)
    norm_factor = 6 * 21 # ensure score in interval (-1, 1)
    p0_score = p1_score = 0
    for y, row in enumerate(board):
        for cell in row:
            if cell == "0":
                p0_score += (ROWS - y)
            elif cell == "1":
                p1_score += (ROWS - y)
    return (p0_score - p1_score) / norm_factor

def eval2(board):
    return threeOutOfFour(board)

def threeOutOfFour(board):
    # p0's 3/4's are counted positively, and p1's 3/4's are counted negatively
    count = 0

    def all_same(lst):
        return all(x == lst[0] for x in lst)

    # Check horizontal chunks
    deltas = [(0, 0), (0, 1), (0, 2), (0, 3)]
    for row in range(6):
        for col in range(4):
            window = []
            for dy, dx in deltas:
                window.append(board[row + dy][col + dx])
            for i in range(4):
                three_window = window[:i] + window[i + 1:]
                if window[i] == 0 and three_window[0] != 0 and all_same(three_window):
                    count += 1 if three_window[0] == 1 else -1

    # Check vertical chunks
    deltas = [(0, 0), (1, 0), (2, 0), (3, 0)]
    for row in range(3):
        for col in range(7):
            window = []
            for dy, dx in deltas:
                window.append(board[row + dy][col + dx])
            for i in range(4):
                three_window = window[:i] + window[i + 1:]
                if window[i] == 0 and three_window[0] != 0 and all_same(three_window):
                    count += 1 if three_window[0] == 1 else -1

    # Check forward diagonal chunks
    deltas = [(0, 0), (1, 1), (2, 2), (3, 3)]
    for row in range(3):
        for col in range(4):
            window = []
            for dy, dx in deltas:
                window.append(board[row + dy][col + dx])
            for i in range(4):
                three_window = window[:i] + window[i + 1:]
                if window[i] == 0 and three_window[0] != 0 and all_same(three_window):
                    count += 1 if three_window[0] == 1 else -1

    # Check backward diagonal chunks
    deltas = [(0, 0), (1, -1), (2, -2), (3, -3)]
    for row in range(3):
        for col in range(4):
            window = []
            for dy, dx in deltas:
                window.append(board[row + dy][col + dx])
            for i in range(4):
                three_window = window[:i] + window[i + 1:]
                if window[i] == 0 and three_window[0] != 0 and all_same(three_window):
                    count += 1 if three_window[0] == 1 else -1
    return count
