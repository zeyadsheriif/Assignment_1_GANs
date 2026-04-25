# Assignment_1_GANs

Medical Image Representation Learning: AE vs VAE
================================================

 What is this project?
------------------------

Modern machine learning systems rely heavily on the ability to learn compact and meaningful representations of complex data. This project explores the design, implementation, and analytical comparison of **Autoencoders (AE)** and **Variational Autoencoders (VAE)** for unsupervised learning, dimensionality reduction, and generative modeling within the domain of medical imaging.

Specifically, this repository tackles representation learning using the **Medical MNIST** dataset. To prevent latent space entanglement and isolate the generative features of distinct medical textures, the training pipeline dynamically builds, trains, and evaluates separate models for each of the 6 distinct anatomical regions.

This project was developed for **DSAI 490**.

Key Features
--------------

*   **Denoising Autoencoder (DAE):** Implements a convolutional encoder-decoder architecture specifically adapted to reconstruct clean images from inputs corrupted with 20% Gaussian noise, testing the model's robustness and utility for artifact removal.
    
*   **Variational Autoencoder (VAE):** Features a probabilistic latent space utilizing the reparameterization trick, optimized via a joint Mean Squared Error (MSE) reconstruction loss and Kullback-Leibler (KL) Divergence penalty.
    
*   **Region-Specific Modeling:** Automated isolation of training pipelines for distinct anatomical classes (AbdomenCT, BreastMRI, CXR, ChestCT, Hand, HeadCT).
    
*   **Optimized Data Pipelines:** Utilizes the tf.data API with aggressive caching and prefetching (AUTOTUNE) for highly efficient data loading and on-the-fly tensor transformations.
    
*   **Latent Space Visualization:** Projects 16-dimensional latent vectors into 2D space using Principal Component Analysis (PCA) to map intra-class variations.
    
*   **Synthetic Sample Generation:** Hallucinates entirely novel medical scans by sampling from the VAE's learned standard normal prior $\\mathcal{N}(0, I)$.
    

 Repository Structure
-----------------------

Plaintext

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   ├── Assignment_Code.ipynb     # Main experiment notebook (Data pipeline, Models, Training, Viz)  ├── Technical_Report.pdf      # Detailed 2-page analysis of architectures, latent spaces, and results  ├── README.md                 # Project documentation   `

Dataset
----------

The **Medical MNIST** dataset is utilized for this experiment. The pipeline is configured to automatically fetch the latest version programmatically via kagglehub (andrewmvd/medical-mnist). It consists of 6 distinct categories:

1.  AbdomenCT
    
2.  BreastMRI
    
3.  CXR (Chest X-Ray)
    
4.  ChestCT
    
5.  Hand
    
6.  HeadCT
    

 Prerequisites & Setup
------------------------

To ensure strict reproducibility, it is recommended to run this project within a virtual environment (such as Conda) or via containerization (Docker).

**1\. Clone the repository:**

Bash

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   git clone https://github.com/yourusername/medical-representation-learning.git  cd medical-representation-learning   `

**2\. Install dependencies:**

Bash

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   pip install tensorflow numpy matplotlib scikit-learn pillow kagglehub   `

 Usage
--------

The codebase is modular and fully optimized for execution in **Google Colab** or locally via **Visual Studio Code** utilizing GPU acceleration.

1.  Open Assignment\_Code.ipynb.
    
2.  Ensure your environment is utilizing a GPU (In Colab: Runtime > Change runtime type > T4 GPU).
    
3.  Execute the cells sequentially.
    
4.  **Note on the Main Loop:** The final execution cell utilizes tf.distribute.MirroredStrategy() and automatically clears the TensorFlow backend session (tf.keras.backend.clear\_session()) between training each region to prevent memory leaks and graph retracing bottlenecks.
    

 Evaluation Metrics Tracked
-----------------------------

During execution, the following metrics are visualized per epoch:

*   **DAE:** Mean Squared Error (MSE) / Denoising Loss
    
*   **VAE:** Total Loss, Reconstruction Loss (MSE), and KL Divergence Loss
