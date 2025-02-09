#include "../cts_helper.cuh"

#define DSIZE 16384
#define block_size 256

int main() {
  auto h_a = mp::make_buffer<float>(DSIZE, DSIZE);
  auto h_b = mp::make_buffer<float>(DSIZE);
  auto d_a = mp::make_buffer<float, mp::device::cuda>(DSIZE, DSIZE);
  auto d_b = mp::make_buffer<float, mp::device::cuda>(DSIZE);
  mp::par::seq().run(h_a.shape(),
                     [a = h_a.view(), b = h_b.view()](const auto &idx) {
                       a(idx) = 1.0f;
                       b[idx[0]] = 0.0f;
                     });

  { // Copy
    mp::copy(d_a.view(), h_a.view());
    mp::copy(d_b.view(), h_b.view());
  }

  auto grids = mp::make_shape(DSIZE / block_size);
  auto blocks = mp::make_shape(block_size);

  cts_begin(row_mp, 10);
  mp::par::cuda().run(
      grids, blocks,
      [a = d_a.const_view(), b = d_b.view()] __device__(auto bid, auto tid) {
        auto i = bid[0] * block_size + tid[0];
        float sum = 0.0f;
        for (int j = 0; j < DSIZE; j++) {
          sum += a(i, j);
        }
        b[i] = sum;
      });
  cts_end(row_mp); // 12~13ms

  cts_begin(col_mp, 10);
  mp::par::cuda().run(
      grids, blocks,
      [a = d_a.const_view(), b = d_b.view()] __device__(auto bid, auto tid) {
        auto i = bid[0] * block_size + tid[0];
        auto sum = 0.0f;
        for (int j = 0; j < DSIZE; j++) {
          sum += a(j, i);
        }
        b[i] = sum;
      });
  cts_end(col_mp); // 3~4ms

  mp::copy(h_b.view(), d_b.view());
  mp::par::seq().run(h_b.shape(), [b = h_b.view()](const auto &idx) {
    if (b(idx) != DSIZE) {
      std::cerr << "Mismatch at " << idx << ": " << b(idx) << std::endl;
      exit(EXIT_FAILURE);
    }
  });

  return EXIT_SUCCESS;
}