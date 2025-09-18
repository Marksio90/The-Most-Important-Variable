# frontend/progress_tracker.py
class ProgressTracker:
    def __init__(self):
        self.progress_bar = st.progress(0)
        self.status_text = st.empty()
        self.steps = []
    
    def add_step(self, name, weight=1):
        self.steps.append({'name': name, 'weight': weight, 'completed': False})
    
    def complete_step(self, step_name):
        for step in self.steps:
            if step['name'] == step_name:
                step['completed'] = True
                break
        self._update_progress()
    
    def _update_progress(self):
        completed_weight = sum(s['weight'] for s in self.steps if s['completed'])
        total_weight = sum(s['weight'] for s in self.steps)
        progress = completed_weight / total_weight if total_weight > 0 else 0
        
        self.progress_bar.progress(progress)
        current_step = next((s['name'] for s in self.steps if not s['completed']), "Ukończone!")
        self.status_text.text(f"Krok: {current_step}")