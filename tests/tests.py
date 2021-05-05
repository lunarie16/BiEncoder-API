import unittest

import torch
from src.biencodernel.knn import FaissHNSWIndex, FaissExactKNNIndex


class FaissHNSWIndexTest(unittest.TestCase):
    def setUp(self) -> None:
        self.test_vectors = {
            'a': torch.tensor([1.0, 1.0, 1.0]),
            'b': torch.tensor([1.0, 1.0, 0.9]),
            'c': torch.tensor([1.0, 1.0, 0.8]),
            'd': torch.tensor([1.0, 1.0, 0.7]),
            'e': torch.tensor([1.0, 1.0, 0.6]),
            'f': torch.tensor([1.0, 1.0, 0.5]),
            'g': torch.tensor([1.0, 1.0, 0.4]),
            'h': torch.tensor([1.0, 1.0, 0.3])
        }
        self.index = FaissHNSWIndex(self.test_vectors)

    def test_len_k1(self):
        query = self.test_vectors['a']
        self.assertEqual(1, len(self.index.get_knn_ids_for_vector(query, k=1)))

    def test_len_k5(self):
        query = self.test_vectors['a']
        self.assertEqual(5, len(self.index.get_knn_ids_for_vector(query, k=5)))

    def test_len_max(self):
        query = self.test_vectors['a']
        self.assertEqual(8, len(self.index.get_knn_ids_for_vector(query, k=500)))

    def test_self(self):
        query = self.test_vectors['e']
        ids, similarities = zip(*self.index.get_knn_ids_for_vector(query, k=1))
        ids = [id[0] for id in ids]
        self.assertEqual(['e'], ids)

    def test_self_2d_shape(self):
        query = self.test_vectors['e'].view(1, -1)
        ids, similarities = zip(*self.index.get_knn_ids_for_vector(query, k=1))
        ids = [id[0] for id in ids]
        self.assertEqual(['e'], ids)

    def test_neighbours(self):
        query = self.test_vectors['e']
        ids, distances = zip(*self.index.get_knn_ids_for_vector(query, k=3))
        ids = set(ids)
        self.assertEqual({'e', 'd', 'f'}, ids)

    def test_distance_order(self):
        query = self.test_vectors['a']
        _, distances = zip(*self.index.get_knn_ids_for_vector(query, k=3))
        self.assertGreater(distances[1], distances[0])
        self.assertGreater(distances[2], distances[1])

class FaissExactKNNIndexTest(unittest.TestCase):
    def setUp(self) -> None:
        self.test_vectors = {
            'a': torch.tensor([1.0, 1.0, 1.0]),
            'b': torch.tensor([1.0, 1.0, 0.9]),
            'c': torch.tensor([1.0, 1.0, 0.8]),
            'd': torch.tensor([1.0, 1.0, 0.7]),
            'e': torch.tensor([1.0, 1.0, 0.6]),
            'f': torch.tensor([1.0, 1.0, 0.5]),
            'g': torch.tensor([1.0, 1.0, 0.4]),
            'h': torch.tensor([1.0, 1.0, 0.3])
        }
        self.index = FaissExactKNNIndex(self.test_vectors)

    def test_len_k1(self):
        query = self.test_vectors['a']
        self.assertEqual(1, len(self.index.get_knn_ids_for_vector(query, k=1)))

    def test_len_k5(self):
        query = self.test_vectors['a']
        self.assertEqual(5, len(self.index.get_knn_ids_for_vector(query, k=5)))

    def test_len_max(self):
        query = self.test_vectors['a']
        self.assertEqual(8, len(self.index.get_knn_ids_for_vector(query, k=500)))

    def test_self(self):
        query = self.test_vectors['e']
        ids, similarities = zip(*self.index.get_knn_ids_for_vector(query, k=1))
        ids = [id[0] for id in ids]
        self.assertEqual(['e'], ids)

    def test_self_2d_shape(self):
        query = self.test_vectors['e'].view(1, -1)
        ids, similarities = zip(*self.index.get_knn_ids_for_vector(query, k=1))
        ids = [id[0] for id in ids]
        self.assertEqual(['e'], ids)

    def test_neighbours(self):
        query = self.test_vectors['e']
        ids, distances = zip(*self.index.get_knn_ids_for_vector(query, k=3))
        ids = set(ids)
        self.assertEqual({'e', 'd', 'f'}, ids)


if __name__ == '__main__':
    unittest.main()