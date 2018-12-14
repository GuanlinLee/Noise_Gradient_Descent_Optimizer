import tensorflow as tf

max_grad_norm=5.0

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
	opt=tf.train.GradientDescentOptimizer(learning_rate)
	grads_and_vars = opt.compute_gradients(loss)
  	noise_grads_and_vars=[(tf.squeeze(tf.clip_by_global_norm([gv[0]],max_grad_norm)[0],axis=0)+tf.nn.l2_normalize(tf.random_normal(gv[0].shape.as_list(),0.0,1.0),dim=0),gv[1]) for gv in grads_and_vars]
  	train_step_key_pred_net = opt.apply_gradients(noise_grads_and_vars)
