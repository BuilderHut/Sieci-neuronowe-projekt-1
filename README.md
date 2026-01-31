Instrukcja uruchomienia (Windows + Conda)
1) Wejdź do folderu z plikiem
2) Aktywuj środowisko conda conda activate ...
3) Uruchomienia A–E (jak w sprawozdaniu)
   
A — baseline
python xor_mlp_onefile.py --early_stop --target_mse 1e-3 --target_clf_err 0.0 --outdir outputs/run_A_baseline


B — momentum
python xor_mlp_onefile.py --momentum 0.9 --early_stop --target_mse 1e-3 --target_clf_err 0.0 --outdir outputs/run_B_momentum


C — adaptive LR (bold driver) + rollback
python xor_mlp_onefile.py --adaptive_lr --rollback_on_worse --early_stop --target_mse 1e-3 --target_clf_err 0.0 --outdir outputs/run_C_adaptive_lr


D — mini-batch (SGD)
python xor_mlp_onefile.py --batch_size 1 --early_stop --target_mse 1e-3 --target_clf_err 0.0 --outdir outputs/run_D_minibatch1


E — momentum + adaptive LR + mini-batch
python xor_mlp_onefile.py --momentum 0.9 --adaptive_lr --rollback_on_worse --batch_size 1 --early_stop --target_mse 1e-3 --target_clf_err 0.0 --outdir outputs/run_E_all


5) Po każdym uruchomieniu powstaną wykresy w odpowiednim folderze outputs/run_*/:

- mse_and_layer_errors.png
- classification_error.png
- learning_rate.png
- weights_layer1.png
- weights_layer2.png
- oraz plik logs.npz.
