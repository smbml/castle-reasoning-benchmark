import numpy as np

import tests.helpers.column_collection as column_collection
import tests.helpers.qualification_columns as qualification_columns


def test_unsupported_terminal_cube_is_rejected():
    model, qpos = column_collection.make_model(2)
    target = qualification_columns.sample_target(19, 0, 2)
    qpos[14:].reshape(2, 7)[:, :3] = target.positions
    hold = np.broadcast_to(qpos, (51, model.nq)).copy()
    assert qualification_columns.contact_report(model, hold, target.positions, target.supports)[
        "passed"
    ]
    top = target.removal_order[0]
    hold[:, 14 + 7 * top + 2] += 0.01
    report = qualification_columns.contact_report(model, hold, target.positions, target.supports)
    assert not report["passed"]
    assert any("missing_support" in reason for reason in report["failures"])
