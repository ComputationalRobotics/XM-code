#pragma once
#include <cstdint>
#include <iostream>
#include <unordered_map>
#include <vector>
#include <bitset>
namespace glomap {

// UnionFind class to maintain disjoint sets for creating tracks
template <typename DataType>
class UnionFind {
 public:
  static constexpr uint32_t num_image = 20000;
  // Find the root of the element x
  DataType Find(DataType x) {
    // If x is not in parent map, initialize it with x as its parent
    auto parentIt = parent_.find(x);
    if (parentIt == parent_.end()) {
      std::bitset<num_image> overlap_element;
      uint32_t image_id = static_cast<uint32_t>(x >> 32);
      overlap_element.set(image_id);
      parent_.emplace_hint(parentIt, x, x);
      overlap.emplace_hint(overlap.end(), x, overlap_element);
      return x;
    }
    // Path compression: set the parent of x to the root of the set containing x
    if (parentIt->second != x) {

      parentIt->second = Find(parentIt->second);
    }
    return parentIt->second;
  }

  void Union(DataType x, DataType y) {
      DataType root_x = Find(x);
      DataType root_y = Find(y);
      if (root_x == root_y) return;

      auto& overlap1 = overlap.find(root_x)->second;
      auto& overlap2 = overlap.find(root_y)->second;

      // Calculate overlap using bitwise AND
      std::bitset<num_image> overlap_result = overlap1 & overlap2;
      size_t overlap_sum = overlap_result.count(); // Count the number of set bits

      if (overlap_sum < 1) {
        // Union by setting the parent
        parent_[root_x] = root_y;

        // Calculate the union of the overlaps using bitwise OR
        overlap2 |= overlap1;
      }
    }


  void Clear() { parent_.clear(); overlap.clear();}

 private:
  // Map to store the parent of each element
  std::unordered_map<DataType, DataType> parent_;
  std::unordered_map<DataType, std::bitset<num_image>> overlap;
};

}  // namespace glomap