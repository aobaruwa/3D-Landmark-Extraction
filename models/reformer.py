import gin
import trax
trax.fastmath.use_backend('jax')
from trax.fastmath import numpy as jnp
from trax.supervised import training 
from trax import layers as tl
import os
import sys

batch_size = 8


def _XY():
  """Returns a layer that computes the element-wise average of two arrays."""
  return tl.Fn('XYAvg', lambda x, y: (x + y) )

def _dot():
  return tl.Fn('XYdot', lambda x, y: jnp.dot(x,y))

def _RemoveAxes12():
  """Returns a layer that removes two internal size-1 axes from an array."""
  return tl.Fn('RemoveAxes12', lambda x: jnp.squeeze(x, (1, 2)))

def _PaddingMask():
  """Returns a layer that maps integer sequences to padding masks.
  Args:
    pad: Integer that represents padding rather than a token/content ID.
  """
  def f(x):
    sequence_length = x.shape[1]
    content_positions = (jnp.sum(x,axis=-1) != 0)
    return content_positions.reshape((batch_size, sequence_length))
  return tl.Fn('PaddingMask()', f)
 
def _AttentionMask():
  def f(q,k,v,m):
    batch_size = q.shape[0]
    sequence_length = k.shape[1]
    content_positions = (jnp.sum(k,axis=-1) != 0)
    mask=content_positions.reshape((batch_size, 1,1,sequence_length))
    return q,k,v,mask
  return tl.Fn('AttMask',f,n_out=4)


def prepare_batch(example,size=12800):
      in_seq=example[0][0:size] #This has been shuffled beforehand
      data= [in_seq,
             jnp.array(jnp.sum(in_seq,axis=-1) !=0),
             jnp.array(example[1]),
             jnp.array(example[2]),
             jnp.array(example[2] !=0) ]
      return data


#Have to pad to a multiple of 128 so it won't fail
def gen(xy_pairs,size):
  while True:
    for example in xy_pairs:
      data=prepare_batch(example,size)
      yield data

@gin.configurable
def noamLR(step_num, d_model=128, warmup_steps=5e2, factor = 1.0):   # todo : make d_model, warmup_steps global/config params
    lr = factor * (
            d_model ** (-0.5) * min(step_num ** (-0.5), step_num * warmup_steps ** (-1.5))
        )
    return lr

@gin.configurable    # include func args in the config.gin file
def build_model(n_heads = 5, 
                n_enc_layers = 3,
                n_dec_layers = 1,
                vocab_size = 162,
                d_model =128,
                d_ff =50):
  
    encoder_blocks=[]
    for i in range(4):
      encoder_blocks+=[
                       tl.Branch([],
                                 [tl.research.efficient_attention.LSHSelfAttention(mode='train', use_reference_code=False,masked=True,n_heads=5,d_qk=d_model,d_v=d_model,chunk_len=256),
                                 ],
                                 [tl.Select([1])]),       
                       _XY(), #Att,mask
                        tl.LayerNorm(),

                        tl.Dense(d_model),
                        tl.Relu(),
                       tl.LayerNorm(),

      ]

    encoder=tl.Serial(encoder_blocks)

    enc_dec_attention = tl.AttentionQKV(n_heads=2,d_feature=d_model)

    model = tl.Serial(
                      tl.Dense(d_model),
                      tl.Relu(),                                        # e m t l lm
                      encoder,                                        # e  m t l lm
                      tl.Select([2,0,1],n_in=3),                      # t e mask l lm
                      tl.Embedding(vocab_size=162,d_feature=d_model), # dt e mask l lm
                      tl.Branch([
                        tl.Select([0,1,1,2]),
                        _AttentionMask(),
                        enc_dec_attention,
                        tl.Select([0],n_in=2)] #Remove Mask
                                ,[]),
                      _XY(),
                      tl.Dense(d_model),
                      tl.Relu(),
                      tl.Dense(3)
      )
    
    return model


if __name__=="__main__":
  # Data loading
    data_dir = sys.argv[1]
    size=int(sys.argv[3])
    print('Stat Data Loading')
    train_data = jnp.load(os.path.join(data_dir, 'train.npy'),allow_pickle=True)
    eval_data = jnp.load(os.path.join(data_dir, 'val.npy'),allow_pickle=True)
    

    train_stream = gen(train_data,size)
    eval_stream = gen(eval_data,size)

    print('Begin Pipeline Building')

    data_pipeline = trax.data.Serial(trax.data.Shuffle(),
                                     trax.data.Batch(batch_size)
                                    )

    eval_pipeline = trax.data.Serial(
                                     trax.data.Batch(batch_size)
                                    )

    train_batches_stream = data_pipeline(train_stream)
    eval_batches_stream = eval_pipeline(eval_stream)

    print('Create Example Batch')
    example_batch =next(train_batches_stream)
    eval_batch = next(eval_batches_stream)
    print(f'batch shape (point_cloud, tokens, label, mask) = {[x.shape for x in example_batch]}')
    print(f'batch shape (point_cloud, tokens, label, mask) = {[x.shape for x in eval_batch]}')
    eb=[jnp.array(i) for i in example_batch]

    ### Modeling

    model=build_model()

    model.init(trax.shapes.signature(eb))

    train_task = training.TrainTask(
        labeled_data=train_batches_stream,
        loss_layer=tl.metrics.L2Loss(),
        optimizer=trax.optimizers.Adam(1e-5),
        lr_schedule=noamLR,
        n_steps_per_checkpoint=500#//batch_size,
    )

    # Evaluaton task.
    eval_task = training.EvalTask(
        labeled_data=eval_batches_stream,
        metrics=[tl.metrics.L2Loss()],
        n_eval_batches=1  # For less variance in eval numbers.
    )

    # Training loop saves checkpoints to output_dir.
    print('A thinks with the loop')
    output_dir = sys.argv[2]+"_"+str(size)
    training_loop = training.Loop(model,
                                  train_task,
                                  eval_tasks=[eval_task],
                                  output_dir=output_dir
    )

    print('Loop?')
    training_loop.run(10000000)
