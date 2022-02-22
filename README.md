
# InsuranceRecommender

This repository includes the code base used in the paper "Evaluation of Algorithms for Interaction-Sparse Recommendations: Neural Networks don't Always Win", accepted at EDBT 2022. Please use the following text to cite our work:

> Yasamin Klingler, Claude Lehmann, João Pedro Monteiro, Carlo Saladin, Abraham Bernstein, and Kurt Stockinger. 2022. Evaluation of Algorithms for Interaction-Sparse Recommendations: Neural Networks don’t Always Win. In *International Conference on Extending Database Technology* (EDBT).


### Authors
* Yasamin Klingler; Zurich University of Applied Sciences; Winterthur, Switzerland; [yasamin.klingler@zhaw.ch](yasamin.klingler@zhaw.ch)
* Claude Lehmann; Zurich University of Applied Sciences; Winterthur, Switzerland; [claude.lehmann@zhaw.ch](claude.lehmann@zhaw.ch)
* João Pedro Monteiro; Veezoo AG; Zurich, Switzerland; [jp@veezoo.com](jp@veezoo.com)
* Carlo Saladin; Veezoo AG; Zurich, Switzerland; [carlo@veezoo.com](carlo@veezoo.com)
* Abraham Bernstein; University of Zurich; Zurich, Switzerland; [bernstein@ifi.uzh.ch](bernstein@ifi.uzh.ch)
* Kurt Stockinger; Zurich University of Applied Sciences; Winterthur, Switzerland; [kurt.stockinger@zhaw.ch](kurt.stockinger@zhaw.ch)

### Algorithm Implementations
- **Alternating Least Squares (ALS)**
based on the ALS implementation of [LibRecommender](https://github.com/massquantity/LibRecommender/blob/master/libreco/algorithms/als.py).
- **DeepFM**
based on the DeepFM implementation of [LibRecommender](https://github.com/massquantity/LibRecommender/blob/master/libreco/algorithms/deepfm.py).
- **Joint Collaborative Autoencoder (JCA)**
based on the JCA implementation of [Ziwei Zhu](https://github.com/Zziwei/Joint-Collaborative-Autoencoder) (original paper implementation).
- **Neural Collaborative Filtering (NCF)**
based on the NCF/NeuMF implementation of [LibRecommender](https://github.com/massquantity/LibRecommender/blob/master/libreco/algorithms/ncf.py).
- **Popularity**
implemented ourselves.
- **Singular Value Decomposition (SVD++)**
based on the SVD++ implementation of [LibRecommender](https://github.com/massquantity/LibRecommender/blob/master/libreco/algorithms/svdpp.py)

### MovieLens1M-Max5 and -Min6
The modified versions of the popular MovieLens1M dataset (and data splits used in the paper mentioned above) can be found here: https://drive.switch.ch/index.php/s/DuAbnp69AUyjA8w
