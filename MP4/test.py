def main():
    state = [(1,2), (2,3), (5,8)]
    p = (1,2)
    state = [item for item in state if item != p]
    print(state)


main()