#![feature(int_roundings)]
#![feature(int_log)]

#[cfg(test)]
use iterx::Iterx;
#[cfg(test)]
use rust_tx::{build_vector, TensorIntOps, TensorOps, TensorResult};

// SST = scan scan transform
// (not really parallel)
#[cfg(test)]
pub fn sum_scan_sst(vec: Vec<i32>) -> Vec<i32> {
    let tile_size = if vec.len() < 32 { 4 } else { 32 };

    // scan
    let tiles = vec
        .chunks(tile_size)
        .map(|tile| tile.iter().copied().scan_(|x, y| x + y));

    // scan
    let sums = tiles
        .clone()
        .into_iter()
        .map(|tile| tile.last().unwrap())
        .prescan(0, |x, y| x + y)
        .drop_last();

    // transform
    tiles
        .zip_map(sums, |tile, sum| tile.map(move |x| x + sum))
        .flatten()
        .collect()
}

#[cfg(test)]
fn sum_scan_sst_with_tx(vec: Vec<i32>) -> TensorResult<i32> {
    let tile_size: i32 = if vec.len() < 32 { 4 } else { 32 };

    // scan
    let tiles = build_vector(vec)
        .chunk(tile_size as usize)?
        .scan(|x, y| x + y, Some(2))?;

    // scan
    let sums = tiles
        .clone()
        .last(Some(2))?
        .prescan(0, |x, y| x + y, None)?
        .drop_last(None)?;

    // transform
    Ok(tiles.plus(sums)?.flatten())
}

// is something like this even possible / useful?
// fn sum_scan_SST_ONE_CHAIN(vec: Vec<i32>) -> Vec<i32> {
//     let tile_size = 32;

//     let tiles = vec
//         .windows(tile_size)
//         // .parallel()
//         .map(|tile| tile.iter().scan_(Add)) // flat_map?
//         .prescan_fuse(|tile| tile.last(), 0, Add)
//         .flatten()
//         .collect();
// }

// RSS = reduce scan scan
// (not really parallel)
#[cfg(test)]
pub fn sum_scan_rss(vec: Vec<i32>) -> Vec<i32> {
    let tile_size = if vec.len() < 32 { 4 } else { 32 };

    // reduce
    let tile_sums = vec
        .chunks(tile_size)
        .map(|tile| tile.iter().copied().fold(0, |x, y| x + y));

    // scan
    let sums = tile_sums.clone().prescan(0, |x, y| x + y).drop_last();

    // transform
    vec.chunks(tile_size)
        .zip_map(sums, |tile, sum| {
            tile.iter().copied().prescan(sum, |x, y| x + y).skip(1)
        })
        .flatten()
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    use itertools::assert_equal;

    #[test]
    fn test_scan_sst() {
        assert_equal(
            (1..=8).into_iter().scan_(|x, y| x + y),
            sum_scan_sst((1..=8).collect()),
        );
        assert_equal(
            (1..=64).into_iter().scan_(|x, y| x + y),
            sum_scan_sst((1..=64).collect()),
        );
    }

    #[test]
    fn test_scan_sst_with_tx() {
        assert_eq!(
            (1..=8).into_iter().scan_(|x, y| x + y).collect::<Vec<_>>(),
            sum_scan_sst_with_tx((1..=8).collect())
                .unwrap()
                .to_vec()
                .unwrap(),
        );
        assert_eq!(
            (1..=64).into_iter().scan_(|x, y| x + y).collect::<Vec<_>>(),
            sum_scan_sst_with_tx((1..=64).collect())
                .unwrap()
                .to_vec()
                .unwrap(),
        );
    }

    #[test]
    fn test_scan_rss() {
        assert_eq!(
            (1..=8).into_iter().scan_(|x, y| x + y).collect::<Vec<_>>(),
            sum_scan_rss((1..=8).collect()),
        );
        assert_equal(
            (1..=64).into_iter().scan_(|x, y| x + y),
            sum_scan_rss((1..=64).collect()),
        );
    }
}
