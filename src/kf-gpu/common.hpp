#pragma once

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace kfgpu {

constexpr double kProcessNoise = 0.01;
constexpr double kMeasurementNoise = 5.0;
constexpr double kInitialCovariance = 500.0;

struct NpyArray {
  std::vector<int64_t> shape;
  std::vector<double> data;
};

struct ExperimentData {
  int64_t filter_count = 0;
  int64_t step_count = 0;
  int64_t state_dim = 0;
  std::vector<double> initial_states;
  std::vector<double> measurements;
};

struct Paths {
  std::string initial_states = "initial_states.npy";
  std::string measurements = "measurements.npy";
  std::string reference = "reference_outputs.npy";
  std::string output_dir = "outputs";
};

inline int64_t shape_product(const std::vector<int64_t>& shape) {
  return std::accumulate(
      shape.begin(),
      shape.end(),
      int64_t{1},
      [](int64_t lhs, int64_t rhs) { return lhs * rhs; });
}

inline std::vector<int64_t> parse_shape(const std::string& header) {
  const auto shape_key = header.find("'shape':");
  if (shape_key == std::string::npos) {
    throw std::runtime_error("NPY header is missing shape.");
  }

  const auto open = header.find('(', shape_key);
  const auto close = header.find(')', open);
  if (open == std::string::npos || close == std::string::npos || close <= open) {
    throw std::runtime_error("NPY shape tuple is malformed.");
  }

  std::vector<int64_t> shape;
  std::stringstream shape_stream(header.substr(open + 1, close - open - 1));
  std::string token;
  while (std::getline(shape_stream, token, ',')) {
    token.erase(
        std::remove_if(
            token.begin(),
            token.end(),
            [](unsigned char value) { return std::isspace(value) != 0; }),
        token.end());
    if (!token.empty()) {
      shape.push_back(std::stoll(token));
    }
  }

  if (shape.empty()) {
    throw std::runtime_error("NPY shape tuple is empty.");
  }

  return shape;
}

inline std::string extract_descr(const std::string& header) {
  const auto key = header.find("'descr':");
  if (key == std::string::npos) {
    throw std::runtime_error("NPY header is missing descr.");
  }

  const auto first_quote = header.find('\'', key + 8);
  const auto second_quote = header.find('\'', first_quote + 1);
  if (first_quote == std::string::npos || second_quote == std::string::npos) {
    throw std::runtime_error("NPY descr field is malformed.");
  }

  return header.substr(first_quote + 1, second_quote - first_quote - 1);
}

inline void ensure_c_order(const std::string& header) {
  const auto key = header.find("'fortran_order':");
  if (key == std::string::npos || header.find("False", key) == std::string::npos) {
    throw std::runtime_error("Only C-order NPY arrays are supported.");
  }
}

inline std::string read_npy_header(std::ifstream& input) {
  char magic[6];
  input.read(magic, sizeof(magic));
  if (!input || std::string(magic, sizeof(magic)) != "\x93NUMPY") {
    throw std::runtime_error("File is not a valid NPY array.");
  }

  unsigned char version[2];
  input.read(reinterpret_cast<char*>(version), sizeof(version));
  if (!input) {
    throw std::runtime_error("Failed to read NPY version.");
  }

  uint32_t header_len = 0;
  if (version[0] == 1) {
    unsigned char bytes[2];
    input.read(reinterpret_cast<char*>(bytes), sizeof(bytes));
    if (!input) {
      throw std::runtime_error("Failed to read NPY v1 header length.");
    }
    header_len = static_cast<uint32_t>(bytes[0]) |
                 (static_cast<uint32_t>(bytes[1]) << 8U);
  } else {
    unsigned char bytes[4];
    input.read(reinterpret_cast<char*>(bytes), sizeof(bytes));
    if (!input) {
      throw std::runtime_error("Failed to read NPY v2/v3 header length.");
    }
    header_len = static_cast<uint32_t>(bytes[0]) |
                 (static_cast<uint32_t>(bytes[1]) << 8U) |
                 (static_cast<uint32_t>(bytes[2]) << 16U) |
                 (static_cast<uint32_t>(bytes[3]) << 24U);
  }

  std::string header(header_len, '\0');
  input.read(header.data(), static_cast<std::streamsize>(header.size()));
  if (!input) {
    throw std::runtime_error("Failed to read NPY header.");
  }

  return header;
}

inline NpyArray load_npy_f64(const std::string& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    throw std::runtime_error("Unable to open " + path);
  }

  const std::string header = read_npy_header(input);
  ensure_c_order(header);

  const std::string descr = extract_descr(header);
  if (descr != "<f8" && descr != "|f8") {
    throw std::runtime_error("Only little-endian float64 NPY arrays are supported.");
  }

  NpyArray array;
  array.shape = parse_shape(header);
  array.data.resize(static_cast<size_t>(shape_product(array.shape)));
  input.read(
      reinterpret_cast<char*>(array.data.data()),
      static_cast<std::streamsize>(array.data.size() * sizeof(double)));
  if (!input) {
    throw std::runtime_error("Failed to read NPY payload from " + path);
  }

  return array;
}

inline ExperimentData load_experiment(
    const std::string& initial_states_path,
    const std::string& measurements_path) {
  const NpyArray initial_states = load_npy_f64(initial_states_path);
  const NpyArray measurements = load_npy_f64(measurements_path);

  if (initial_states.shape.size() != 2) {
    throw std::runtime_error("initial_states.npy must have shape (N, state_dim).");
  }
  if (measurements.shape.size() != 3) {
    throw std::runtime_error("measurements.npy must have shape (N, T, state_dim).");
  }
  if (initial_states.shape[0] != measurements.shape[0]) {
    throw std::runtime_error("Initial state batch size does not match measurements.");
  }
  if (initial_states.shape[1] != measurements.shape[2]) {
    throw std::runtime_error("State dimension does not match measurement dimension.");
  }

  ExperimentData experiment;
  experiment.filter_count = initial_states.shape[0];
  experiment.step_count = measurements.shape[1];
  experiment.state_dim = initial_states.shape[1];
  experiment.initial_states = initial_states.data;
  experiment.measurements = measurements.data;
  return experiment;
}

inline std::string make_npy_header(const std::vector<int64_t>& shape) {
  std::ostringstream shape_stream;
  shape_stream << "(";
  for (size_t index = 0; index < shape.size(); ++index) {
    if (index != 0) {
      shape_stream << ", ";
    }
    shape_stream << shape[index];
  }
  if (shape.size() == 1) {
    shape_stream << ",";
  }
  shape_stream << ")";

  std::string header =
      "{'descr': '<f8', 'fortran_order': False, 'shape': " +
      shape_stream.str() +
      ", }";

  const size_t preamble = 10;
  const size_t aligned = ((preamble + header.size() + 1 + 15) / 16) * 16;
  header.append(aligned - preamble - header.size() - 1, ' ');
  header.push_back('\n');
  return header;
}

inline void write_npy_f64(
    const std::string& path,
    const std::vector<int64_t>& shape,
    const std::vector<double>& data) {
  if (shape_product(shape) != static_cast<int64_t>(data.size())) {
    throw std::runtime_error("NPY shape does not match data size.");
  }

  std::filesystem::create_directories(std::filesystem::path(path).parent_path());

  std::ofstream output(path, std::ios::binary);
  if (!output) {
    throw std::runtime_error("Unable to open output file " + path);
  }

  const std::string header = make_npy_header(shape);
  const char magic[] = "\x93NUMPY";
  output.write(magic, 6);
  const unsigned char version[2] = {1, 0};
  output.write(reinterpret_cast<const char*>(version), 2);
  const uint16_t header_len = static_cast<uint16_t>(header.size());
  output.write(reinterpret_cast<const char*>(&header_len), sizeof(header_len));
  output.write(header.data(), static_cast<std::streamsize>(header.size()));
  output.write(
      reinterpret_cast<const char*>(data.data()),
      static_cast<std::streamsize>(data.size() * sizeof(double)));
  if (!output) {
    throw std::runtime_error("Failed to write output file " + path);
  }
}

inline void write_raw_final_states(
    const std::string& path,
    const std::vector<double>& trajectories,
    int64_t filter_count,
    int64_t step_count,
    int64_t state_dim) {
  const size_t final_offset =
      static_cast<size_t>((step_count - 1) * state_dim);

  std::vector<double> final_states(static_cast<size_t>(filter_count * state_dim));
  for (int64_t filter_index = 0; filter_index < filter_count; ++filter_index) {
    const size_t src =
        static_cast<size_t>(filter_index * step_count * state_dim) + final_offset;
    const size_t dst = static_cast<size_t>(filter_index * state_dim);
    std::copy_n(trajectories.data() + src, state_dim, final_states.data() + dst);
  }

  std::filesystem::create_directories(std::filesystem::path(path).parent_path());
  std::ofstream output(path, std::ios::binary);
  if (!output) {
    throw std::runtime_error("Unable to open output file " + path);
  }

  output.write(reinterpret_cast<const char*>(&filter_count), sizeof(filter_count));
  output.write(reinterpret_cast<const char*>(&state_dim), sizeof(state_dim));
  output.write(
      reinterpret_cast<const char*>(final_states.data()),
      static_cast<std::streamsize>(final_states.size() * sizeof(double)));
  if (!output) {
    throw std::runtime_error("Failed to write raw output file " + path);
  }
}

inline void copy_reference_output(
    const std::string& reference_path,
    const std::string& output_dir) {
  std::filesystem::create_directories(output_dir);
  std::filesystem::copy_file(
      reference_path,
      std::filesystem::path(output_dir) / "reference_outputs.npy",
      std::filesystem::copy_options::overwrite_existing);
}

inline Paths parse_paths(int argc, char** argv) {
  Paths paths;
  if (argc >= 2) {
    paths.initial_states = argv[1];
  }
  if (argc >= 3) {
    paths.measurements = argv[2];
  }
  if (argc >= 4) {
    paths.reference = argv[3];
  }
  if (argc >= 5) {
    paths.output_dir = argv[4];
  }
  return paths;
}

inline size_t state_offset(int64_t filter_index, int64_t state_dim) {
  return static_cast<size_t>(filter_index * state_dim);
}

inline size_t measurement_offset(
    int64_t filter_index,
    int64_t step_index,
    int64_t step_count,
    int64_t state_dim) {
  return static_cast<size_t>((filter_index * step_count + step_index) * state_dim);
}

}  // namespace kfgpu
