from pynsfem.utils.power_tuples import power_tuples_sum, power_tuples_max


class TestPowerTuplesSum:
    def test_single_dimension(self):
        """Test power_tuples_sum for n=1 with various powers"""
        assert list(power_tuples_sum(1, 0)) == [(0,)]
        assert list(power_tuples_sum(1, 1)) == [(1,)]
        assert list(power_tuples_sum(1, 2)) == [(2,)]
        assert list(power_tuples_sum(1, 3)) == [(3,)]

    def test_two_dimensions(self):
        """Test power_tuples_sum for n=2 with various powers"""
        assert sorted(list(power_tuples_sum(2, 0))) == [(0, 0)]
        assert sorted(list(power_tuples_sum(2, 1))) == [(0, 1), (1, 0)]
        assert sorted(list(power_tuples_sum(2, 2))) == [(0, 2), (1, 1), (2, 0)]
        assert sorted(list(power_tuples_sum(2, 3))) == [(0, 3), (1, 2), (2, 1), (3, 0)]

    def test_three_dimensions(self):
        """Test power_tuples_sum for n=3 with various powers"""
        assert sorted(list(power_tuples_sum(3, 0))) == [(0, 0, 0)]
        assert sorted(list(power_tuples_sum(3, 1))) == [(0, 0, 1), (0, 1, 0), (1, 0, 0)]
        assert sorted(list(power_tuples_sum(3, 2))) == [
            (0, 0, 2),
            (0, 1, 1),
            (0, 2, 0),
            (1, 0, 1),
            (1, 1, 0),
            (2, 0, 0),
        ]

    def test_sum_property(self):
        """Test that all generated tuples sum to the specified power"""
        for n in range(1, 5):
            for power in range(4):
                for tup in power_tuples_sum(n, power):
                    assert sum(tup) == power, f"Sum of {tup} should be {power}"

    def test_count(self):
        """Test the number of generated tuples matches the combinatorial formula"""
        for n in range(1, 5):
            for power in range(5):
                # Formula for counting n-tuples that sum to power: (n+power-1) choose (n-1)
                from math import comb

                expected_count = comb(n + power - 1, n - 1)
                actual_count = len(list(power_tuples_sum(n, power)))
                assert actual_count == expected_count


class TestPowerTuplesMax:
    def test_single_dimension(self):
        """Test power_tuples_max for n=1 with various powers"""
        assert list(power_tuples_max(1, 0)) == [(0,)]
        assert list(power_tuples_max(1, 1)) == [(1,)]
        assert list(power_tuples_max(1, 2)) == [(2,)]

    def test_two_dimensions(self):
        """Test power_tuples_max for n=2 with various powers"""
        assert sorted(list(power_tuples_max(2, 0))) == [(0, 0)]
        assert sorted(list(power_tuples_max(2, 1))) == [(0, 1), (1, 0), (1, 1)]
        assert sorted(list(power_tuples_max(2, 2))) == [
            (0, 2),
            (1, 2),
            (2, 0),
            (2, 1),
            (2, 2),
        ]

    def test_three_dimensions(self):
        """Test power_tuples_max for n=3 with power=1"""
        expected = [
            (0, 0, 1),
            (0, 1, 0),
            (0, 1, 1),
            (1, 0, 0),
            (1, 0, 1),
            (1, 1, 0),
            (1, 1, 1),
        ]
        assert sorted(list(power_tuples_max(3, 1))) == expected

    def test_max_property(self):
        """Test that all generated tuples have a maximum value equal to the specified power"""
        for n in range(1, 4):
            for power in range(3):
                for tup in power_tuples_max(n, power):
                    assert max(tup) == power, f"Max of {tup} should be {power}"

    def test_count_small_cases(self):
        """Test counts for small cases with known values"""
        # For n=1: Should have exactly 1 tuple for any power
        for power in range(4):
            assert len(list(power_tuples_max(1, power))) == 1

        # For power=0: Should have exactly 1 tuple for any n
        for n in range(1, 5):
            assert len(list(power_tuples_max(n, 0))) == 1
