#!/usr/bin/env python3
"""Inspect a joblib-saved sklearn model and print useful attributes.

Usage:
    python inspect_model.py path/to/model.joblib
"""
import sys
import joblib
import pprint


def main():
    if len(sys.argv) < 2:
        print("Usage: python inspect_model.py path/to/model.joblib")
        sys.exit(1)

    path = sys.argv[1]
    print(f"Loading model: {path}")
    model = joblib.load(path)
    print("Type:", type(model))

    # Common attributes
    attrs = {}
    for a in ('n_features_in_', 'feature_names_in_', 'classes_', 'n_outputs_'):
        if hasattr(model, a):
            attrs[a] = getattr(model, a)

    # If it's a pipeline, show steps
    try:
        from sklearn.pipeline import Pipeline
        if isinstance(model, Pipeline):
            attrs['is_pipeline'] = True
            attrs['pipeline_steps'] = [(name, type(step)) for name, step in model.steps]
        else:
            attrs['is_pipeline'] = False
    except Exception:
        attrs['is_pipeline'] = 'sklearn not available'

    # Show parameter keys (top-level)
    try:
        params = model.get_params()
        attrs['param_keys'] = list(params.keys())[:50]
    except Exception as e:
        attrs['param_keys_error'] = str(e)

    pprint.pprint(attrs)


if __name__ == '__main__':
    main()
