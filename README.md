# GPDN


---

## 🚀 Training

Step.1 Download the pretrained diffusion models and StyleGAN

Create a folder ./diffusion/models/ and download model checkpoints into it

Download FFHQ or CelebA-HQ with [P2-weighting](https://github.com/jychoi118/P2-weighting).

Create a folder ./pretrained

Download StyleGANv2 checkpoints from [mmagic](https://github.com/open-mmlab/mmagic)

Step.2 Training
Two optional ways to start training (run at the project root directory):

```bash
bash train.sh
```

or directly use the python code

```bash
python tools/train.py
```

---

This project is built based on **mmedit** and **GLEAN** frameworks.
