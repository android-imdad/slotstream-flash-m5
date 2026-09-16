"""Offline tests of objective task scoring and counterbalanced schedules only."""
import csv
import io
import json
import unittest
import tasks_run as driver


class QualityTests(unittest.TestCase):
    def setUp(self):
        self.tasks = {t['id']: t for t in json.loads((driver.FIXTURES / 'tasks.json').read_text())['tasks']}

    def test_frozen_json_answers_and_surrounding_whitespace(self):
        for task in self.tasks.values():
            if task['kind'] == 'json':
                self.assertTrue(driver.quality(task, ' \n' + json.dumps(task['expected']) + '\n ', 'stop')['passed'])

    def test_reject_duplicate_keys(self):
        self.assertFalse(driver.quality(self.tasks['short-json'], '{"sum":0,"sum":51}', 'stop')['passed'])
        self.assertFalse(driver.quality(self.tasks['short-json'], '{"sum":51,"sum":51}', 'stop')['passed'])

    def test_boolean_integer_and_float_types(self):
        boolean = self.tasks['grounded-document']
        changed = dict(boolean['expected'], serve_has_widening_flag=0)
        self.assertFalse(driver.quality(boolean, json.dumps(changed), 'stop')['passed'])
        self.assertFalse(driver.quality(self.tasks['short-json'], '{"sum":51.0}', 'stop')['passed'])
        self.assertFalse(driver.typed_equal([True], [1]))

    def test_truncation_even_when_parsed_answer_is_correct(self):
        self.assertFalse(driver.quality(self.tasks['short-json'], '{"sum":51}', 'length')['passed'])
        self.assertFalse(driver.quality(self.tasks['short-json'], '{"sum":', 'stop')['passed'])

    def test_json_fences_prose_extra_keys_wrong_value(self):
        for answer in ('```json\n{"sum":51}\n```', '{"sum":51}\nDone', '{"sum":51,"extra":0}', '{"sum":52}', '{"sum":NaN}'):
            self.assertFalse(driver.quality(self.tasks['short-json'], answer, 'stop')['passed'], answer)

    def csv_text(self, rows):
        output = io.StringIO()
        csv.writer(output).writerows(rows)
        return output.getvalue()

    def test_csv_full_frozen_table(self):
        task = self.tasks['long-csv']
        self.assertEqual(len(task['expected']), 33)
        self.assertTrue(driver.quality(task, '\n ' + self.csv_text(task['expected']) + '\n ', 'stop')['passed'])

    def test_csv_reject_wrong_values_rows_headers_fences(self):
        task = self.tasks['long-csv']
        rows = task['expected']
        wrong = [r[:] for r in rows]
        wrong[17][1] = '288'
        variants = [rows[:-1], rows + [rows[-1]], [['n','cube','square']] + rows[1:], wrong]
        for variant in variants:
            self.assertFalse(driver.quality(task, self.csv_text(variant), 'stop')['passed'])
        self.assertFalse(driver.quality(task, '```csv\n' + self.csv_text(rows) + '```', 'stop')['passed'])
        self.assertFalse(driver.quality(task, self.csv_text(rows) + 'Done', 'stop')['passed'])


class NumericalCohortTests(unittest.TestCase):
    def test_exact_plan020_cohort_and_wrong_candidate_or_run_set(self):
        valid = {'run_set_id': 'prefill-512-qualification-20260916',
                 'reference_chunk': 256, 'empirical_control_chunk': 384, 'candidate_chunk': 512}
        driver.validate_numeric_cohort(valid)
        for key, wrong in (('run_set_id', 'prefill-qualification-20260916'),
                           ('candidate_chunk', 1024), ('empirical_control_chunk', 512),
                           ('reference_chunk', 256.0)):
            with self.assertRaises(driver.EvidenceError, msg=key):
                driver.validate_numeric_cohort(valid | {key: wrong})


class ScheduleTests(unittest.TestCase):
    def test_screen_has_each_task_chunk_once(self):
        rows = list(driver.schedule('screen'))
        self.assertEqual(len(rows), 8)
        self.assertEqual({(r['task'], r['chunk']) for r in rows}, {(t,c) for t in driver.TASK_IDS for c in driver.CHUNKS})
        self.assertTrue(all(r['round'] == 0 for r in rows))
        self.assertEqual([rows[i]['chunk'] for i in (0,2,4,6)], [256,512,256,512])

    def test_extension_has_balanced_three_round_positions(self):
        screen = list(driver.schedule('screen'))
        extension = list(driver.schedule('extend'))
        self.assertEqual(len(extension), 8)
        self.assertEqual({r['task'] for r in extension}, set(driver.EXTEND_IDS))
        for task in driver.EXTEND_IDS:
            for chunk in driver.CHUNKS:
                rows = [r for r in screen + extension if r['task'] == task and r['chunk'] == chunk]
                self.assertEqual([r['round'] for r in rows], [0,1,2])
                positions = [r['order'].index(chunk) for r in rows]
                self.assertEqual(positions[0], positions[2])
                self.assertNotEqual(positions[0], positions[1])
        for chunk in driver.CHUNKS:
            positions = [r['order'].index(chunk) for r in screen + extension if r['chunk'] == chunk]
            self.assertEqual(positions.count(0), 4)
            self.assertEqual(positions.count(1), 4)
        self.assertEqual(len(screen + extension), 16)


if __name__ == '__main__':
    unittest.main()
