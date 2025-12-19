from pathlib import Path
from typing import Optional, Dict
import h5py
import torch
from config.config import VisualizationConfig
from models import ABMIL

class ABMIL_Visualizer:
    def __init__(self, config: VisualizationConfig, wsi_path: str):
        self.config = config
        self.wsi_path = wsi_path
        self.wsi_name = Path(wsi_path).stem
        self.job_dir = Path(self.config.job_dir) / self.wsi_name
        self.job_dir.mkdir(parents=True, exist_ok=True)
        self.abmil_weights_path = self.config.abmil_weights_path
        self.device = self.config.device

        # WSI loading
        self.wsi = self._load_wsi()

        # Pipeline state tracking
        self._geojson_contours = None
        self._coords_path = None
        self._features_path = None
        self._coords = None
        self._coords_attrs = None
        self._patch_features = None
        self._logits = None
        self._attention_scores = None

        # Models (lazy loading)
        self._segmentation_model = None
        self._patch_encoder = None
        self._abmil_encoder = None

    def _load_wsi(self):
        from trident.wsi_objects.WSIFactory import load_wsi
        print("Loading WSI.....")
        self.wsi = load_wsi(slide_path=self.wsi_path)
        print("Done!")
        return self.wsi

    @property
    def segmentation_model(self):
        from trident.segmentation_models import segmentation_model_factory
        if self._segmentation_model is None:
            self._segmentation_model = segmentation_model_factory(
                self.config.segmentation_model_name
            )
        return self._segmentation_model

    @property
    def patch_encoder(self):
        from trident.patch_encoder_models import encoder_factory as patch_encoder_factory
        if self._patch_encoder is None:
            self._patch_encoder = patch_encoder_factory(
                self.config.patch_encoder_name
            )
        return self._patch_encoder

    @property
    def abmil_encoder(self):
        if self._abmil_encoder is None:
            self._abmil_encoder = ABMIL(
                input_dim=self.config.input_dim,
                hidden_dim=self.config.hidden_dim,
                num_classes=self.config.num_classes,
                n_heads=self.config.n_heads,
                dropout=self.config.dropout,
                gated=self.config.gated
            ).to(self.device)
        return self._abmil_encoder

    def segment_tissue(self,
                       recompute: bool = False):
        if self._geojson_contours is not None and not recompute:
            return self._geojson_contours

        print("Segmenting tissues...")
        self._geojson_contours = self.wsi.segment_tissue(
            segmentation_model = self.segmentation_model,
            job_dir = str(self.job_dir),
            device = self.device
        )
        print(f"Segmentation results saved to {str(self.job_dir)}")
        return self._geojson_contours

    def extract_coords(self,
                       target_mag: Optional[int] = None,
                       patch_size: Optional[int] = None,
                       overlap: Optional[int] = None,
                       recompute: bool = False):

        if self._coords_path is not None and not recompute:
            return self._coords_path

        target_mag = target_mag or self.config.target_mag
        patch_size = patch_size or self.config.patch_size
        overlap = overlap or self.config.overlap

        print("Extracting patch coords...")
        self._coords_path = self.wsi.extract_tissue_coords(
            target_mag=target_mag,
            patch_size=patch_size,
            save_coords=str(self.job_dir),
            overlap=overlap,
        )
        print(f"Coords saved to: {self._coords_path}")

        return self._coords_path

    def extract_features(self,
                         coords_path: Optional[str] = None,
                         recompute: bool = False):

        coords_path = coords_path or self._coords_path

        if coords_path is None:
            raise RuntimeError(
                "Coordinates not available. Run extract_coords() first or provide coords_path."
            )

        features_dir = self.job_dir / f"features_{self.config.patch_encoder_name}"

        if self._features_path is not None and not recompute:
            if Path(self._features_path).exists():
                return self._features_path
            else:
                raise RuntimeError(
                    "The raw features has been cleaned up!"
                )

        print(f"Extracting patch features with {self.config.patch_encoder_name} ...")
        self._features_path = self.wsi.extract_patch_features(
            patch_encoder=self.patch_encoder,
            coords_path=coords_path,
            save_features=str(features_dir),
            device=self.device,
        )
        print(f"Features saved to: {self._features_path}")

        return self._features_path

    def load_features(self,
                      features_path: Optional[str] = None):

        features_path = features_path or self._features_path
        if features_path is None:
            raise RuntimeError(
                "Features path not available. Run extract_features() first or provide features_path."
            )

        print(f"Loading features from: {features_path}")

        with h5py.File(features_path, 'r') as f:
            self._coords = f['coords'][:]
            self._patch_features = f['features'][:]
            self._coords_attrs = dict(f['coords'].attrs)
        print(f"Loaded {len(self._patch_features)} patch features")

        return self._patch_features, self._coords, self._coords_attrs

    def run_inference(self,
                      features_path: Optional[str] = None):

        if self._patch_features is None:
            self.load_features(features_path)

        print("Running ABMIL inference...")

        # Run model
        with torch.no_grad():
            self._logits, self._attention_scores = self.abmil_encoder(
                torch.from_numpy(self._patch_features).float().to(self.device).unsqueeze(0)
            )

        print(f"Inference completed. Logits shape: {self._logits.shape}")
        return self._logits, self._attention_scores

    def visualize(self,
                  vis_level: Optional[int] = None,
                  num_top_patches: Optional[int] = None,
                  normalize: Optional[bool] = None,
                  output_dir: Optional[str] = None):

        from trident import visualize_heatmap
        if self._attention_scores is None:
            raise RuntimeError(
                "Attention scores not available. Run run_inference() first or provide scores."
            )
        scores = self._attention_scores

        if self._coords is None or self._coords_attrs is None:
            raise RuntimeError(
                "Coordinates not loaded. Run load_features() or extract_features() first."
            )

        # Use config defaults if not specified
        vis_level = vis_level if vis_level is not None else self.config.vis_level
        num_top_patches = num_top_patches if num_top_patches is not None else self.config.num_top_patches
        normalize = normalize if normalize is not None else self.config.normalize_heatmap
        output_dir = output_dir or str(self.job_dir)

        print(f"Generating heatmap for {self.wsi_name}...")
        heatmap_path = visualize_heatmap(
            wsi=self.wsi,
            scores=scores.cpu().numpy().squeeze() if torch.is_tensor(scores) else scores.squeeze(),
            coords=self._coords,
            vis_level=vis_level,
            patch_size_level0=self._coords_attrs['patch_size_level0'],
            normalize=normalize,
            num_top_patches_to_save=num_top_patches,
            output_dir=output_dir,
        )

        print(f"Heatmap saved to: {heatmap_path}")
        return heatmap_path

    def run_pipeline(
            self,
            coords_kwargs: Optional[Dict] = None,
            features_kwargs: Optional[Dict] = None,
            inference_kwargs: Optional[Dict] = None,
            visualization_kwargs: Optional[Dict] = None,
            recompute: bool = False
    ) -> str:
        print("=" * 60)
        print("Starting ABMIL Visualization Pipeline")
        print("=" * 60)

        # Step 1: Segment tissue
        self.segment_tissue(
            recompute=recompute
        )

        # Step 2: Extract coordinates
        self.extract_coords(
            recompute=recompute,
            **(coords_kwargs or {})
        )

        # Step 3: Extract features
        self.extract_features(
            recompute=recompute,
            **(features_kwargs or {})
        )

        # Step 4: Run inference
        self.run_inference(**(inference_kwargs or {}))

        # Step 5: Visualize
        heatmap_path = self.visualize(**(visualization_kwargs or {}))

        print("=" * 60)
        print("Pipeline completed successfully!")
        print("=" * 60)

        return heatmap_path

    def get_state(self):
        return {
            'slide_path': str(self.wsi_path),
            'slide_name': self.wsi_name,
            'job_dir': str(self.job_dir),
            'device': str(self.device),
            'has_segmentation': self._geojson_contours is not None,
            'coords_path': self._coords_path,
            'features_path': self._features_path,
            'has_inference_results': self._attention_scores is not None,
            'num_patches': len(self._coords) if self._coords is not None else 0,
        }

    def reset_state(self):
        self._geojson_contours = None
        self._coords_path = None
        self._features_path = None
        self._coords = None
        self._coords_attrs = None
        self._patch_features = None
        self._logits = None
        self._attention_scores = None
        print("Pipeline state reset")


if __name__ == "__main__":
    from config.config import visualization_config
    wsi_path = '/mnt/gml/GML/Project/WSI/HE/MALT-Lymphoma/B22-55434-B.sdpc'

    visualizer = ABMIL_Visualizer(config=visualization_config, wsi_path=wsi_path)

    visualizer.segment_tissue()
    visualizer.extract_coords()
    visualizer.extract_features()
    visualizer.run_inference()
    visualizer.visualize()







