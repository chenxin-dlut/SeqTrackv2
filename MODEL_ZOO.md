# Model Zoo

Here we provide the performance of the SeqTrack and SeqTrackv2 trackers on multiple tracking benchmarks and the corresponding raw results. 
The model weights and the corresponding training logs are also given by the links.

## SeqTrack Models

<table>
  <tr>
    <th>Model</th>
    <th>LaSOT<br>AUC (%)</th>
    <th>LaSOT-ext<br>AUC (%)</th>
    <th>GOT-10k<br>AO (%)</th>
    <th>TrackingNet<br>AUC (%)</th>
    <th>VOT2020-bbox<br>EAO</th>
    <th>VOT2020-mask<br>EAO</th>
    <th>TNL2K<br>AUC (%)</th>
    <th>NFS<br>AUC (%)</th>
    <th>UAV<br>AUC (%)</th>
    <th>Models</th>
    <th>Logs</th>
  </tr>
  <tr>
    <td>SeqTrack-B256</td>
    <td>69.9</td>
    <td>49.5</td>
    <td>74.7</td>
    <td>83.3</td>
    <td>29.1</td>
    <td>52.0</td>
    <td>54.9</td>
    <td>67.6</td>
    <td>69.2</td>
    <td><a href="https://drive.google.com/drive/folders/10PKs6aqSbVtb6aloYmtaN4aKMnmPF58P?usp=sharing">[Google]</a><a href="https://pan.baidu.com/s/16_wrhpHPwa9D8eyUCbJzqA?pwd=iiau">[Baidu]</a></td>
    <td><a href="https://drive.google.com/drive/folders/1_kf2noaP3M9RHHCgG6XqBFejNkzBv2xB?usp=sharing">log</a></td>
  </tr>
  <tr>
    <td>SeqTrack-B384</td>
    <td>71.5</td>
    <td>50.5</td>
    <td>74.5</td>
    <td>83.9</td>
    <td>31.2</td>
    <td>52.2</td>
    <td>56.4</td>
    <td>66.7</td>
    <td>68.6</td>
    <td><a href="https://drive.google.com/drive/folders/10PKs6aqSbVtb6aloYmtaN4aKMnmPF58P?usp=sharing">[Google]</a><a href="https://pan.baidu.com/s/16_wrhpHPwa9D8eyUCbJzqA?pwd=iiau">[Baidu]</a></td>
    <td><a href="https://drive.google.com/drive/folders/1_kf2noaP3M9RHHCgG6XqBFejNkzBv2xB?usp=sharing">log</a></td> 
  </tr>
  <tr>
    <td>SeqTrack-L256</td>
    <td>72.1</td>
    <td>50.5</td>
    <td>74.5</td>
    <td>85.0</td>
    <td>31.3</td>
    <td>55.5</td>
    <td>56.9</td>
    <td>66.9</td>
    <td>69.7</td>
    <td><a href="https://drive.google.com/drive/folders/10PKs6aqSbVtb6aloYmtaN4aKMnmPF58P?usp=sharing">[Google]</a><a href="https://pan.baidu.com/s/16_wrhpHPwa9D8eyUCbJzqA?pwd=iiau">[Baidu]</a></td>
    <td><a href="https://drive.google.com/drive/folders/1_kf2noaP3M9RHHCgG6XqBFejNkzBv2xB?usp=sharing">log</a></td>
  </tr>
  <tr>
    <td>SeqTrack-L384</td>
    <td>72.5</td>
    <td>50.7</td>
    <td>74.8</td>
    <td>85.5</td>
    <td>31.9</td>
    <td>56.1</td>
    <td>57.8</td>
    <td>66.2</td>
    <td>68.5</td>
    <td><a href="https://drive.google.com/drive/folders/10PKs6aqSbVtb6aloYmtaN4aKMnmPF58P?usp=sharing">[Google]</a><a href="https://pan.baidu.com/s/16_wrhpHPwa9D8eyUCbJzqA?pwd=iiau">[Baidu]</a></td>
    <td><a href="https://drive.google.com/drive/folders/1_kf2noaP3M9RHHCgG6XqBFejNkzBv2xB?usp=sharing">log</a></td>
  </tr>
</table>

The downloaded checkpoints should be organized in the following structure
   ```
   ${SeqTrack_ROOT}
    -- checkpoints
        -- train
            -- seqtrack
                -- seqtrack_b256
                    SEQTRACK_ep0500.pth.tar
                -- seqtrack_b256_got
                    SEQTRACK_ep0500.pth.tar
                -- seqtrack_b384
                    SEQTRACK_ep0500.pth.tar
                -- seqtrack_b384_got
                    SEQTRACK_ep0500.pth.tar
                -- seqtrack_l256
                    SEQTRACK_ep0500.pth.tar
                -- seqtrack_l256_got
                    SEQTRACK_ep0500.pth.tar
                -- seqtrack_l384
                    SEQTRACK_ep0500.pth.tar
                -- seqtrack_l384_got
                    SEQTRACK_ep0500.pth.tar
   ```

## SeqTrackv2 Models

<table>
  <tr>
    <th>Model</th>
    <th>LasHeR<br>AUC (%)</th>
    <th>RGBT234<br>MSR (%)</th>
    <th>VOT-RGBD22<br>EAO (%)</th>
    <th>DepthTrack<br>F-score (%)</th>
    <th>VisEvent<br>AUC</th>
    <th>TNL2K<br>AUC</th>
    <th>OTB99<br>AUC (%)</th>
    <th>Models</th>
  </tr>
  <tr>
    <td>SeqTrackv2-B256</td>
    <td>55.8</td>
    <td>64.7</td>
    <td>74.4</td>
    <td>63.2</td>
    <td>61.2</td>
    <td>57.5</td>
    <td>71.2</td>
    <td><a href="https://drive.google.com/drive/folders/1G9dfrETn6szp2Kxli8c8KCZHgbciVDVB?usp=sharing">[Google]</a><a href="https://pan.baidu.com/s/16_wrhpHPwa9D8eyUCbJzqA?pwd=iiau">[Baidu]</a></td>
  </tr>
  <tr>
    <td>SeqTrackv2-B384</td>
    <td>56.2</td>
    <td>66.3</td>
    <td>75.5</td>
    <td>59.8</td>
    <td>62.2</td>
    <td>59.4</td>
    <td>71.8</td>
    <td><a href="https://drive.google.com/drive/folders/1G9dfrETn6szp2Kxli8c8KCZHgbciVDVB?usp=sharing">[Google]</a><a href="https://pan.baidu.com/s/16_wrhpHPwa9D8eyUCbJzqA?pwd=iiau">[Baidu]</a></td>
  </tr>
  <tr>
    <td>SeqTrackv2-L256</td>
    <td>58.8</td>
    <td>68.5</td>
    <td>74.9</td>
    <td>62.8</td>
    <td>63.0</td>
    <td>62.7</td>
    <td>70.3</td>
    <td><a href="https://drive.google.com/drive/folders/1G9dfrETn6szp2Kxli8c8KCZHgbciVDVB?usp=sharing">[Google]</a><a href="https://pan.baidu.com/s/16_wrhpHPwa9D8eyUCbJzqA?pwd=iiau">[Baidu]</a></td>
  </tr>
  <tr>
    <td>SeqTrackv2-L384</td>
    <td>61.0</td>
    <td>68.0</td>
    <td>74.8</td>
    <td>62.3</td>
    <td>63.4</td>
    <td>62.4</td>
    <td>71.4</td>
    <td><a href="https://drive.google.com/drive/folders/1G9dfrETn6szp2Kxli8c8KCZHgbciVDVB?usp=sharing">[Google]</a><a href="https://pan.baidu.com/s/16_wrhpHPwa9D8eyUCbJzqA?pwd=iiau">[Baidu]</a></td>
  </tr>
</table>

The downloaded checkpoints should be organized in the following structure
   ```
   ${SeqTrack_ROOT}
    -- checkpoints
        -- train
            -- seqtrackv2
                -- seqtrackv2_b256
                    SEQTRACKV2_ep0240.pth.tar
                -- seqtrackv2_b384
                    SEQTRACKV2_ep0240.pth.tar
                -- seqtrackv2_l256
                    SEQTRACKV2_ep0240.pth.tar
                -- seqtrackv2_l384
                    SEQTRACKV2_ep0240.pth.tar
   ```

## SeqTrack Raw Results for RGB-based benchmarks
The [raw results](https://drive.google.com/drive/folders/15xrVifqG_idkXVxJOhUWq7nB5rzNxyO_?usp=sharing) are in the format [top_left_x, top_left_y, width, height]. Raw results of GOT-10K and TrackingNet can be 
directly submitted to the corresponding evaluation servers. The folder ```test/tracking_results/``` contains raw results and results should be organized in the following structure
   ```
   ${SeqTrack_ROOT}
    -- test
        -- tracking_results
            -- seqtrack
                -- seqtrack_b256
                    --lasot
                        airplane-1.txt
                        airplane-13.txt
                        ...
                    --lasot_extension_subset
                        atv-1.txt
                        atv-2.txt
                        ...
                -- seqtrack_b384
                    --lasot
                        airplane-1.txt
                        airplane-13.txt
                        ...
                ...
   ```
The raw results of VOT2020 should be organized in the following structure
   ```
   ${SeqTrack_ROOT}
    -- external
        -- vot20
            -- seqtrack
                -- results
                    --seqtrack_b256_ar
                    --seqtrack_b384_ar
                    ...
   ```

## SeqTrackv2 Raw Results for Multi-Modal benchmarks
The [raw results](https://drive.google.com/file/d/1Jrgk_rhw_t1-qboMCuegUYCVyGGNwuv0/view?usp=sharing) are in the format [top_left_x, top_left_y, width, height]. The results should be evaluated using each benchmark's official toolkit.
