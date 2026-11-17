import ast
from importlib import import_module
from pathlib import Path
import subprocess
import sys
import unittest
from cardioagent.common.paths import code_root

class LayoutTests(unittest.TestCase):
    def test_current_controller_import_does_not_load_legacy_or_evaluation(self):
        script="import sys; import cardioagent.ecg_agent.agent; forbidden=('cardioagent.legacy','cardioagent.ecg_agent.evaluation','cardioagent.ecg_agent.benchmarks','cardioagent.ecg_agent.experiments','langchain_openai'); assert not [x for x in sys.modules if any(x==p or x.startswith(p+'.') for p in forbidden)]"
        subprocess.run([sys.executable,'-c',script],check=True)

    def test_online_sources_have_no_reverse_layer_imports(self):
        base=code_root()/'src/cardioagent/ecg_agent'
        paths=[base/'agent.py',base/'schemas.py']
        for name in ('harness','policies','memory','tools','models','data'):paths.extend((base/name).glob('*.py'))
        for path in paths:
            for node in ast.walk(ast.parse(path.read_text(encoding='utf-8'))):
                if isinstance(node,ast.ImportFrom):
                    module=node.module or ''
                    self.assertFalse(any(module.startswith(p) for p in ('cardioagent.legacy','cardioagent.ecg_agent.experiments','cardioagent.ecg_agent.evaluation','cardioagent.ecg_agent.benchmarks')),(path,module))

    def test_old_tool_and_memory_names_are_same_objects(self):
        pairs=[('cardioagent.ecg_agent.toolkit.runtime','cardioagent.ecg_agent.harness.runtime','ToolSession'),
               ('cardioagent.ecg_agent.event_state','cardioagent.ecg_agent.memory.events','EventTracker'),
               ('cardioagent.recording','cardioagent.common.recording','Waveform')]
        for old,new,name in pairs:self.assertIs(getattr(import_module(old),name),getattr(import_module(new),name))

    def test_compatibility_scorer_is_canonical_predictor(self):
        from cardioagent.model_adapters import cached_ecgfounder_scorer
        from cardioagent.ecg_agent.models.ecgfounder import create_ecgfounder_scorer
        self.assertIs(cached_ecgfounder_scorer,create_ecgfounder_scorer)

    def test_vendor_inputs_are_exact_copies_of_previous_model_assets(self):
        root=code_root()
        for name in ('net1d.py','tasks.txt'):
            self.assertEqual((root/'third_party/ecgfounder'/name).read_bytes(),(root/'models/upstream/ecgfounder'/name).read_bytes())

    def test_legacy_cli_and_current_cli_help(self):
        for command in (['--help'],['track-afdb','--help']):
            p=subprocess.run([sys.executable,'-m','cardioagent.cli',*command],capture_output=True,text=True)
            self.assertEqual(p.returncode,0,p.stderr)
            self.assertIn('track-afdb' if command==['--help'] else '--device',p.stdout)
