import sys, collections, math
import openvino as ov
m = ov.Core().read_model(sys.argv[1])
ops = m.get_ordered_ops()
byname = {op.get_friendly_name(): op for op in ops}
# true data dependency: propagate from Parameters but NOT through ShapeOf (static shapes -> foldable)
dep = set()
for op in ops:
    t = op.get_type_name()
    if t == "Parameter":
        dep.add(op.get_friendly_name()); continue
    if t in ("ShapeOf",):
        continue
    for i in range(op.get_input_size()):
        if op.input_value(i).get_node().get_friendly_name() in dep:
            dep.add(op.get_friendly_name()); break
mm = [op for op in ops if op.get_type_name() == "MatMul"]
rows = collections.Counter(); const_mm = []
for op in mm:
    s0 = op.input_value(0).get_partial_shape()
    shp = tuple(s0.to_shape()) if s0.is_static else None
    rows[shp[-2] if shp and len(shp) >= 2 else None] += 1
    if op.get_friendly_name() not in dep:
        const_mm.append(op)
print("MatMul total", len(mm), "| by activation rows (M):", rows.most_common(8))
print("MatMul data-independent of Parameters (ignoring ShapeOf):", len(const_mm))
w = 0
for op in const_mm:
    s1 = op.input_value(1).get_partial_shape()
    w += math.prod(s1.to_shape()) if s1.is_static else 0
print("their weight elements:", w, f"(~{w/1e6:.1f} M; INT8 bytes ~{w/1e6:.1f} MB)")
names = collections.Counter()
for op in const_mm:
    n = op.get_friendly_name()
    names[n.split("/")[0][:60] if "/" in n else n[:60]] += 1
print("sample names:", [op.get_friendly_name()[:100] for op in const_mm[:6]])
# shapes of const matmuls
sh = collections.Counter((tuple(op.input_value(0).get_partial_shape().to_shape()), tuple(op.input_value(1).get_partial_shape().to_shape())) for op in const_mm if op.input_value(0).get_partial_shape().is_static and op.input_value(1).get_partial_shape().is_static)
print("const matmul shapes:", sh.most_common(6))
