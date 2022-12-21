#![feature(int_roundings)]
#![feature(int_log)]

#[cfg(test)]
use iterx::Iterx;
#[cfg(test)]
use rust_tx::{build_vector, TensorIntOps, TensorOps, TensorResult};

// ssm = scan scan transform
// (not really parallel)
#[cfg(test)]
pub fn sum_scan_ssm(vec: Vec<i32>) -> Vec<i32> {
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

    // map
    tiles
        .zip_map(sums, |tile, sum| tile.map(move |x| x + sum))
        .flatten()
        .collect()
}

#[cfg(test)]
fn scan_ssm_with_tx<F>(vec: Vec<i32>, binop: F) -> TensorResult<i32>
where
    F: Fn(&i32, i32) -> i32 + Clone,
{
    let tile_size = if vec.len() < 32 { 4 } else { 32 };

    // scan
    let tiles = build_vector(vec)
        .chunk(tile_size as usize)?
        .scan(&binop, Some(2))?;

    // scan
    let sums = tiles
        .clone()
        .last(Some(2))?
        .prescan(0, binop, None)?
        .drop_last(None)?;

    // map
    Ok(tiles.plus(sums)?.flatten())
}

// is something like this even possible / useful?
// fn sum_scan_ssm_ONE_CHAIN(vec: Vec<i32>) -> Vec<i32> {
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

    // preScan
    vec.chunks(tile_size)
        .zip_map(sums, |tile, sum| {
            tile.iter().copied().prescan(sum, |x, y| x + y).skip(1)
        })
        .flatten()
        .collect()
}

#[cfg(test)]
fn sum_scan_rss_with_tx(vec: Vec<i32>) -> TensorResult<i32> {
    let tile_size = if vec.len() < 32 { 4 } else { 32 };
    let vec_tx = build_vector(vec);

    // reduce
    let tile_sums = vec_tx
        .clone()
        .chunk(tile_size)?
        .reduce(|x, y| x + y, Some(2))?;

    // scan
    let sums = tile_sums.prescan(0, |x, y| x + y, None)?.drop_last(None)?;

    // preScan
    Ok(sums
        .append(vec_tx.chunk(tile_size)?)?
        .scan(|x, y| x + y, Some(2))?
        .drop_first(Some(2))?
        .flatten())
}

#[cfg(test)]
mod tests {
    use super::*;

    use itertools::assert_equal;

    #[test]
    fn test_scan_ssm() {
        assert_equal(
            (1..=8).into_iter().scan_(|x, y| x + y),
            sum_scan_ssm((1..=8).collect()),
        );
        assert_equal(
            (1..=64).into_iter().scan_(|x, y| x + y),
            sum_scan_ssm((1..=64).collect()),
        );
    }

    #[test]
    fn test_scan_ssm_with_tx() {
        assert_eq!(
            (1..=8).into_iter().scan_(|x, y| x + y).collect::<Vec<_>>(),
            scan_ssm_with_tx((1..=8).collect(), |x, y| x + y)
                .unwrap()
                .to_vec()
                .unwrap(),
        );
        assert_eq!(
            (1..=64).into_iter().scan_(|x, y| x + y).collect::<Vec<_>>(),
            scan_ssm_with_tx((1..=64).collect(), |x, y| x + y)
                .unwrap()
                .to_vec()
                .unwrap(),
        );
    }

    #[test]
    fn test_sum_scan_rss() {
        assert_eq!(
            (1..=8).into_iter().scan_(|x, y| x + y).collect::<Vec<_>>(),
            sum_scan_rss((1..=8).collect()),
        );
        assert_equal(
            (1..=64).into_iter().scan_(|x, y| x + y),
            sum_scan_rss((1..=64).collect()),
        );
    }

    #[test]
    fn test_sum_scan_rss_with_tx() {
        assert_eq!(
            (1..=8).into_iter().scan_(|x, y| x + y).collect::<Vec<_>>(),
            sum_scan_rss_with_tx((1..=8).collect())
                .unwrap()
                .to_vec()
                .unwrap(),
        );
        assert_equal(
            (1..=64).into_iter().scan_(|x, y| x + y),
            sum_scan_rss_with_tx((1..=64).collect())
                .unwrap()
                .to_vec()
                .unwrap(),
        );
    }
}
