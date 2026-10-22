from task_aware_detector import TaskAwareDetector

d = TaskAwareDetector()
res = d.select_best_object('sample_kitchen_raw.jpg', 'bake', decompose_task=True)
print('Task:', res['task'], '| Status:', res['status'])
for c in res['all_candidates']:
    print(f" - {c['name']:<12}: Fused={c['fused_score']:.4f}, Sim={c['semantic_similarity']:.4f}, Conf={c['confidence']:.4f}, Valid={c['is_relevant']}")
if res['selected_object']:
    print('>>> SELECTED:', res['selected_object']['name'], 'Score:', res['selected_object']['fused_score'])
else:
    print('>>> NO OBJECT SELECTED')
