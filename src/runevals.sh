# python evaluate.py --model_path ../models/finetune/gpt2_epoch0_ts10000.20210310.18.03.1615401990 --edit_steps 5

# python evaluate.py --model_path ../models/model_epoch0_ts18000.20210315.00.03.1615768579 --edit_steps 1
# python evaluate.py --model_path ../models/model_epoch0_ts18000.20210315.00.03.1615768579 --edit_steps 3
# python evaluate.py --model_path ../models/model_epoch0_ts18000.20210315.00.03.1615768579 --edit_steps 5

# python evaluate.py --model_path ../models/model_epoch0_ts10000.20210314.22.03.1615760445 --edit_steps 1
# python evaluate.py --model_path ../models/model_epoch0_ts10000.20210314.22.03.1615760445 --edit_steps 3
# python evaluate.py --model_path ../models/model_epoch0_ts10000.20210314.22.03.1615760445 --edit_steps 5

# python evaluate.py --model_path ../models/model_epoch0_ts10000.20210312.21.03.1615586239 --edit_steps 1
# python evaluate.py --model_path ../models/model_epoch0_ts10000.20210312.21.03.1615586239 --edit_steps 3
# python evaluate.py --model_path ../models/model_epoch0_ts10000.20210312.21.03.1615586239 --edit_steps 5

# python evaluate.py --model_path ../models/model_epoch0_ts6000.20210314.16.03.1615740151 --edit_steps 1
# python evaluate.py --model_path ../models/model_epoch0_ts6000.20210314.16.03.1615740151 --edit_steps 3
# python evaluate.py --model_path ../models/model_epoch0_ts6000.20210314.16.03.1615740151 --edit_steps 5

# python evaluate.py --model_path ../models/model_epoch0_ts10000.20210314.16.03.1615740151 --edit_steps 1 --test_set
python evaluate.py --model_path ../models/finetune/gpt2_epoch0_ts10000.20210310.18.03.1615401990 --edit_steps 1 --test_set
python evaluate.py --ots --edit_steps 1 --test_set