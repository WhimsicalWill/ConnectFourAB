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

    # Check horizontal chunks
    for row in range(6):
        for col in range(4):
            if board[row][col] != " " and board[row][col] == board[row][col+1] == board[row][col+2]:
                count += 1 if board[row][col] == "0" else -1
            if board[row][col+1] != " " and board[row][col+1] == board[row][col+2] == board[row][col+3]:
                count += 1 if board[row][col] == "0" else -1
            if board[row][col] != " " and board[row][col] == board[row][col+2] == board[row][col+3]:
                count += 1 if board[row][col] == "0" else -1
            if board[row][col] != " " and board[row][col] == board[row][col+1] == board[row][col+3]:
                count += 1 if board[row][col] == "0" else -1

    # Check vertical chunks
    for row in range(3):
        for col in range(7):
            if board[row][col] != "." and board[row][col] == board[row+1][col] == board[row+2][col]:
                count += 1 if board[row][col] == "0" else -1
            if board[row+1][col] != "." and board[row+1][col] == board[row+2][col] == board[row+3][col]:
                count += 1 if board[row][col] == "0" else -1
            if board[row][col] != "." and board[row][col] == board[row+2][col] == board[row+3][col]:
                count += 1 if board[row][col] == "0" else -1
            if board[row][col] != "." and board[row][col] == board[row+1][col] == board[row+3][col]:
                count += 1 if board[row][col] == "0" else -1
    
    # Check forward diagonal chunks
    for row in range(3):
        for col in range(4):
            if board[row][col] != "." and board[row][col] == board[row+1][col+1] == board[row+2][col+2]:
                count += 1 if board[row][col] == "0" else -1
            if board[row+1][col+1] != "." and board[row+1][col+1] == board[row+2][col+2] == board[row+3][col+3]:
                count += 1 if board[row][col] == "0" else -1
            if board[row][col] != "." and board[row][col] == board[row+1][col+1] == board[row+3][col+3]:
                count += 1 if board[row][col] == "0" else -1
            if board[row][col] != "." and board[row][col] == board[row+2][col+2] == board[row+3][col+3]:
                count += 1 if board[row][col] == "0" else -1
    
    # Check backward diagonal chunks
    for row in range(3):
        for col in range(4):
            if board[row][col] != "." and board[row][col] == board[row+1][col-1] == board[row+2][col-2]:
                count += 1 if board[row][col] == "0" else -1
            if board[row+1][col-1] != "." and board[row+1][col-1] == board[row+2][col-2] == board[row+3][col-3]:
                count += 1 if board[row][col] == "0" else -1
            if board[row][col] != "." and board[row][col] == board[row+2][col-2] == board[row+3][col-3]:
                count += 1 if board[row][col] == "0" else -1
            if board[row][col] != "." and board[row][col] == board[row+1][col-1] == board[row+3][col-3]:
                count += 1 if board[row][col] == "0" else -1

    return count / 1000
