"""
Tests for khreibga.board — board geometry, adjacency, rays, and initial setup.
"""

from __future__ import annotations

import pytest

from khreibga.board import (
    ADJACENCY,
    ALL_DIRS,
    BLACK,
    BLACK_KING,
    BLACK_MAN,
    BOARD_SIZE,
    DIAGONAL_DIRS,
    EMPTY,
    NUM_SQUARES,
    ORTHOGONAL_DIRS,
    WHITE,
    WHITE_KING,
    WHITE_MAN,
    display_board,
    get_neighbors,
    get_ray,
    initial_board,
    rc_to_sq,
    sq_to_rc,
)


# ===================================================================
# Coordinate conversion
# ===================================================================

class TestCoordinateConversion:
    """Round-trip and boundary tests for sq_to_rc / rc_to_sq."""

    def test_round_trip_all_squares(self):
        """Converting sq -> (r,c) -> sq should be the identity for 0..24."""
        for sq in range(NUM_SQUARES):
            r, c = sq_to_rc(sq)
            assert rc_to_sq(r, c) == sq

    def test_known_corners(self):
        assert sq_to_rc(0) == (0, 0)
        assert sq_to_rc(4) == (0, 4)
        assert sq_to_rc(20) == (4, 0)
        assert sq_to_rc(24) == (4, 4)

    def test_center(self):
        assert sq_to_rc(12) == (2, 2)
        assert rc_to_sq(2, 2) == 12

    def test_row_col_ranges(self):
        """All generated rows and columns should be in [0, 4]."""
        for sq in range(NUM_SQUARES):
            r, c = sq_to_rc(sq)
            assert 0 <= r < BOARD_SIZE
            assert 0 <= c < BOARD_SIZE


# ===================================================================
# Adjacency structure
# ===================================================================

class TestAdjacency:
    """Verify the ADJACENCY table has the right connectivity."""

    def test_adjacency_length(self):
        assert len(ADJACENCY) == NUM_SQUARES

    # -- Direction counts per square ------------------------------------

    def test_corner_00_directions(self):
        """(0,0): r+c=0 (even) -> has diagonals. 2 ortho + 1 diag = 3 dirs."""
        # ortho: north (1,0), east (0,1)  — south/west are off-board
        # diag: NE (1,1) — the others are off-board
        dirs = ADJACENCY[rc_to_sq(0, 0)]
        assert len(dirs) == 3

    def test_corner_04_directions(self):
        """(0,4): r+c=4 (even) -> diagonals. 2 ortho + 1 diag = 3 dirs."""
        dirs = ADJACENCY[rc_to_sq(0, 4)]
        assert len(dirs) == 3

    def test_corner_40_directions(self):
        """(4,0): r+c=4 (even) -> diagonals. 2 ortho + 1 diag = 3 dirs."""
        dirs = ADJACENCY[rc_to_sq(4, 0)]
        assert len(dirs) == 3

    def test_corner_44_directions(self):
        """(4,4): r+c=8 (even) -> diagonals. 2 ortho + 1 diag = 3 dirs."""
        dirs = ADJACENCY[rc_to_sq(4, 4)]
        assert len(dirs) == 3

    def test_edge_01_directions(self):
        """(0,1): r+c=1 (odd) -> NO diagonals. 3 ortho dirs (N, E, W)."""
        dirs = ADJACENCY[rc_to_sq(0, 1)]
        assert len(dirs) == 3  # north, east, west
        for d in dirs:
            assert d in ORTHOGONAL_DIRS

    def test_edge_10_directions(self):
        """(1,0): r+c=1 (odd) -> NO diagonals. 3 ortho (N, S, E)."""
        dirs = ADJACENCY[rc_to_sq(1, 0)]
        assert len(dirs) == 3
        for d in dirs:
            assert d in ORTHOGONAL_DIRS

    def test_center_22_directions(self):
        """(2,2): r+c=4 (even) -> diagonals. 4 ortho + 4 diag = 8 dirs."""
        dirs = ADJACENCY[rc_to_sq(2, 2)]
        assert len(dirs) == 8

    def test_square_21_directions(self):
        """(2,1): r+c=3 (odd) -> NO diagonals. 4 ortho dirs."""
        dirs = ADJACENCY[rc_to_sq(2, 1)]
        assert len(dirs) == 4
        for d in dirs:
            assert d in ORTHOGONAL_DIRS

    def test_square_02_directions(self):
        """(0,2): r+c=2 (even) -> diagonals. 3 ortho + 2 diag = 5 dirs."""
        # ortho: N, E, W (S is off-board)
        # diag: NE(1,1), NW(1,-1) — the southern diagonals are off-board
        dirs = ADJACENCY[rc_to_sq(0, 2)]
        assert len(dirs) == 5

    # -- Diagonal rule enforcement --------------------------------------

    def test_even_parity_has_diagonals(self):
        """Every square with (r+c) even must have at least one diagonal dir."""
        for sq in range(NUM_SQUARES):
            r, c = sq_to_rc(sq)
            if (r + c) % 2 == 0:
                diag_dirs = [d for d in ADJACENCY[sq] if d in DIAGONAL_DIRS]
                assert len(diag_dirs) >= 1, f"sq {sq} ({r},{c}) missing diags"

    def test_odd_parity_no_diagonals(self):
        """Every square with (r+c) odd must have NO diagonal direction."""
        for sq in range(NUM_SQUARES):
            r, c = sq_to_rc(sq)
            if (r + c) % 2 == 1:
                for d in ADJACENCY[sq]:
                    assert d in ORTHOGONAL_DIRS, (
                        f"sq {sq} ({r},{c}) has unexpected diagonal {d}"
                    )


# ===================================================================
# Ray generation
# ===================================================================

class TestRays:
    """Verify get_ray returns correct ordered sequences."""

    def test_ray_00_north(self):
        """From (0,0) going north (1,0): should be [5, 10, 15, 20]."""
        ray = get_ray(rc_to_sq(0, 0), (1, 0))
        assert ray == [rc_to_sq(1, 0), rc_to_sq(2, 0), rc_to_sq(3, 0), rc_to_sq(4, 0)]
        assert ray == [5, 10, 15, 20]

    def test_ray_00_northeast(self):
        """From (0,0) going NE (1,1): full diagonal [(1,1),(2,2),(3,3),(4,4)]."""
        ray = get_ray(rc_to_sq(0, 0), (1, 1))
        assert ray == [6, 12, 18, 24]

    def test_ray_00_east(self):
        """From (0,0) going east (0,1): [1,2,3,4]."""
        ray = get_ray(rc_to_sq(0, 0), (0, 1))
        assert ray == [1, 2, 3, 4]

    def test_ray_24_southwest(self):
        """From (4,4) going SW (-1,-1): [(3,3),(2,2),(1,1),(0,0)]."""
        ray = get_ray(rc_to_sq(4, 4), (-1, -1))
        assert ray == [18, 12, 6, 0]

    def test_ray_22_all_eight(self):
        """Center (2,2) should have rays in all 8 directions."""
        sq = rc_to_sq(2, 2)
        for d in ALL_DIRS:
            assert len(get_ray(sq, d)) >= 1

    def test_ray_invalid_direction_returns_empty(self):
        """Asking for a diagonal from an odd-parity square returns []."""
        # (0,1) has r+c=1 (odd) -> no diagonals
        sq = rc_to_sq(0, 1)
        for d in DIAGONAL_DIRS:
            assert get_ray(sq, d) == []

    def test_ray_off_board_direction_returns_empty(self):
        """Going south from row 0 should return []."""
        sq = rc_to_sq(0, 2)
        assert get_ray(sq, (-1, 0)) == []

    def test_ray_length_from_corner(self):
        """From (0,0), north ray should have exactly 4 squares."""
        assert len(get_ray(0, (1, 0))) == 4

    def test_ray_length_from_edge_middle(self):
        """From (0,2) going north, ray length should be 4."""
        assert len(get_ray(rc_to_sq(0, 2), (1, 0))) == 4

    def test_ray_consistency_with_adjacency(self):
        """Every ray from get_ray should match the ADJACENCY table exactly."""
        for sq in range(NUM_SQUARES):
            for d, expected_ray in ADJACENCY[sq].items():
                assert get_ray(sq, d) == expected_ray


# ===================================================================
# Neighbors
# ===================================================================

class TestNeighbors:
    """Verify get_neighbors returns the correct immediate neighbours."""

    def test_corner_00_neighbors(self):
        """(0,0) should have exactly 3 neighbors: (0,1), (1,0), (1,1)."""
        nbrs = sorted(get_neighbors(rc_to_sq(0, 0)))
        assert nbrs == sorted([rc_to_sq(0, 1), rc_to_sq(1, 0), rc_to_sq(1, 1)])
        assert nbrs == [1, 5, 6]

    def test_corner_44_neighbors(self):
        """(4,4) should have exactly 3 neighbors: (4,3), (3,4), (3,3)."""
        nbrs = sorted(get_neighbors(rc_to_sq(4, 4)))
        assert nbrs == sorted([rc_to_sq(4, 3), rc_to_sq(3, 4), rc_to_sq(3, 3)])

    def test_center_22_has_8_neighbors(self):
        """(2,2) should have 8 neighbors (4 ortho + 4 diag)."""
        nbrs = get_neighbors(rc_to_sq(2, 2))
        assert len(nbrs) == 8

    def test_center_22_neighbor_values(self):
        expected = sorted([
            rc_to_sq(1, 1), rc_to_sq(1, 2), rc_to_sq(1, 3),
            rc_to_sq(2, 1), rc_to_sq(2, 3),
            rc_to_sq(3, 1), rc_to_sq(3, 2), rc_to_sq(3, 3),
        ])
        assert sorted(get_neighbors(rc_to_sq(2, 2))) == expected

    def test_odd_parity_edge_neighbors(self):
        """(0,1) has r+c=1 (odd) -> only ortho: (0,0), (0,2), (1,1)."""
        nbrs = sorted(get_neighbors(rc_to_sq(0, 1)))
        assert nbrs == sorted([rc_to_sq(0, 0), rc_to_sq(0, 2), rc_to_sq(1, 1)])

    def test_neighbor_count_all_squares(self):
        """Sanity: every square should have between 2 and 8 neighbors."""
        for sq in range(NUM_SQUARES):
            n = len(get_neighbors(sq))
            assert 2 <= n <= 8, f"sq {sq} has {n} neighbors"


# ===================================================================
# Initial board
# ===================================================================

class TestInitialBoard:
    """Verify the starting position matches the Khreibaga spec."""

    @pytest.fixture
    def board(self) -> list[int]:
        return initial_board()

    def test_length(self, board):
        assert len(board) == NUM_SQUARES

    def test_black_count(self, board):
        assert board.count(BLACK_MAN) == 12

    def test_white_count(self, board):
        assert board.count(WHITE_MAN) == 12

    def test_empty_count(self, board):
        assert board.count(EMPTY) == 1

    def test_no_kings_initially(self, board):
        assert board.count(BLACK_KING) == 0
        assert board.count(WHITE_KING) == 0

    def test_center_is_empty(self, board):
        assert board[12] == EMPTY

    def test_black_row0(self, board):
        for sq in range(0, 5):
            assert board[sq] == BLACK_MAN, f"sq {sq} should be BLACK_MAN"

    def test_black_row1(self, board):
        for sq in range(5, 10):
            assert board[sq] == BLACK_MAN, f"sq {sq} should be BLACK_MAN"

    def test_black_row2_left(self, board):
        assert board[10] == BLACK_MAN
        assert board[11] == BLACK_MAN

    def test_white_row2_right(self, board):
        assert board[13] == WHITE_MAN
        assert board[14] == WHITE_MAN

    def test_white_row3(self, board):
        for sq in range(15, 20):
            assert board[sq] == WHITE_MAN, f"sq {sq} should be WHITE_MAN"

    def test_white_row4(self, board):
        for sq in range(20, 25):
            assert board[sq] == WHITE_MAN, f"sq {sq} should be WHITE_MAN"

    def test_row2_center_empty(self, board):
        """Index 12 (row 2, col 2) must be the only empty square."""
        assert board[12] == EMPTY


# ===================================================================
# Display
# ===================================================================

class TestDisplay:
    """Smoke-test the text renderer."""

    def test_display_returns_string(self):
        board = initial_board()
        result = display_board(board)
        assert isinstance(result, str)

    def test_display_contains_pieces(self):
        board = initial_board()
        result = display_board(board)
        assert "b" in result
        assert "w" in result
        assert "." in result

    def test_display_row_order(self):
        """Row 4 should appear before row 0 in the output."""
        board = initial_board()
        result = display_board(board)
        idx4 = result.index("4 |")
        idx0 = result.index("0 |")
        assert idx4 < idx0, "Row 4 should be printed above row 0"


# ===================================================================
# Constants sanity checks
# ===================================================================

class TestConstants:
    def test_board_size(self):
        assert BOARD_SIZE == 5

    def test_num_squares(self):
        assert NUM_SQUARES == 25

    def test_piece_values_distinct(self):
        pieces = {EMPTY, BLACK_MAN, BLACK_KING, WHITE_MAN, WHITE_KING}
        assert len(pieces) == 5

    def test_player_values(self):
        assert BLACK == 1
        assert WHITE == 2


# ===================================================================
# Additional edge cases
# ===================================================================

class TestAdditionalEdgeCases:
    """Extra guardrails for copy semantics and graph stability."""

    def test_get_ray_returns_copy(self):
        """Mutating a returned ray must not affect internal adjacency."""
        sq = rc_to_sq(0, 0)
        ray = get_ray(sq, (1, 0))
        ray.append(999)
        assert get_ray(sq, (1, 0)) == [5, 10, 15, 20]

    def test_initial_board_returns_fresh_list_each_call(self):
        """Each call should return a new board instance."""
        board_a = initial_board()
        board_b = initial_board()
        board_a[0] = EMPTY
        assert board_b[0] == BLACK_MAN

    def test_neighbors_are_unique_for_every_square(self):
        """No square should list duplicate immediate neighbors."""
        for sq in range(NUM_SQUARES):
            nbrs = get_neighbors(sq)
            assert len(nbrs) == len(set(nbrs)), f"Duplicate neighbor in sq {sq}"

    def test_display_contains_column_footer(self):
        """Board display should include the column labels footer."""
        text = display_board(initial_board())
        assert "0  1  2  3  4" in text
