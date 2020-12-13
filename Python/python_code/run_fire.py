import fire


class Calculator(object):
    """A simple calculator class."""

    def double(self, number):
        return 2 * number


if __name__ == '__main__':
    fire.Fire(Calculator)

# Then, from the command line, you can run:

# python calculator.py double 10  # 20
# python calculator.py double --number=15  # 30
