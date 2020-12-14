import sys

if __name__ == "__main__":
    if '--generate' in sys.argv:
        from multiknow_f import generate as main
    else:
        from multiknow_f import main

    main.main()
