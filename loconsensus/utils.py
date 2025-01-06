def offset_indexer(n):
    offset_indices = []
    for i in range(n):
        for j in range(n):
            if j >= i:
                offset_index = (i, j)
                offset_indices.append(offset_index)
    return offset_indices


def row_col_from_cindex(cindex, n):
    r = 0
    while cindex >= (n - r):
        cindex -= n - r
        r += 1
    c = r + cindex
    return r, c
