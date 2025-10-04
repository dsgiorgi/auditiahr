
def sample_size_warning(df, group_col, min_n=30):
    warn=[] 
    for g, gdf in df.groupby(group_col):
        if len(gdf)<min_n: warn.append(f"El grupo '{g}' tiene {len(gdf)} casos (<{min_n}).")
    return warn
