# test_simple_math.py


# A basic test function. Pytest will automatically discover this
# because its name starts with `test_`.
def test_addition():
    """
    This test verifies that the sum of 1 and 1 is 2.
    """
    # The `assert` statement is the core of any test.
    # If the condition is True, the test passes.
    # If the condition is False, the test fails with an AssertionError.
    assert 1 + 1 == 2
