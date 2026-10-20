import { useEffect, useMemo, useRef, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { addReference, assetUrl, thumbUrl, createJob, getJob, listCharacters, type GenerationBody, type HistoryJob } from "./api";
import { listModels } from "./trainingApi";
import { DEFAULT_NEGATIVE, FIELDS, compilePrompt, type Selection } from "./compiler";
import { Alert, Badge, Card, Empty, Field, Icon, PageHead, Segmented, Spinner, type Tone } from "./ui";

const STATE_LABEL: Record<string, { text: string; tone: Tone }> = {
  queued: { text: "Đang chờ worker", tone: "info" },
  running: { text: "Đang tạo ảnh", tone: "primary" },
  review_required: { text: "Hoàn tất — chờ duyệt", tone: "ok" },
  failed: { text: "Thất bại", tone: "bad" },
  cancelled: { text: "Đã hủy", tone: "warn" },
  approved: { text: "Đã duyệt", tone: "ok" },
  rejected: { text: "Bị loại", tone: "bad" },
};

const ASPECTS: { value: GenerationBody["aspect_ratio"]; label: string }[] = [
  { value: "9:16", label: "9:16" },
  { value: "1:1", label: "1:1" },
  { value: "1:1-small", label: "768²" },
  { value: "16:9", label: "16:9" },
];

export function Generate({ reuse }: { reuse?: { n: number; job: HistoryJob } | null }) {
  const [original, setOriginal] = useState("Nhân vật nữ Việt Nam 25 tuổi, tóc đen dài, áo sơ mi trắng, ngồi trong thư viện");
  const [identityEn, setIdentityEn] = useState("portrait of a young Vietnamese woman, long black hair");
  const [sel, setSel] = useState<Selection>({ outfit: "white shirt", scene: "in a library", lighting: "soft window light", style: "cinematic photorealistic photo" });
  const [extraEn, setExtraEn] = useState("");
  const [override, setOverride] = useState<string | null>(null); // người dùng sửa tay prompt đã biên dịch
  const [negative, setNegative] = useState(DEFAULT_NEGATIVE);
  const [aspect, setAspect] = useState<GenerationBody["aspect_ratio"]>("1:1");
  const [num, setNum] = useState(1);
  const [seed, setSeed] = useState(42);
  const [steps, setSteps] = useState(28);
  const [guidance, setGuidance] = useState(6);
  // Nhớ job gần nhất: tải lại trang vẫn thấy kết quả/tiếp tục theo dõi job đang chạy.
  const [jobId, setJobId] = useState<string | null>(() => { try { return localStorage.getItem("emvoox.gen.job"); } catch { return null; } });
  useEffect(() => { try { if (jobId) localStorage.setItem("emvoox.gen.job", jobId); else localStorage.removeItem("emvoox.gen.job"); } catch { /* bỏ qua */ } }, [jobId]);
  const keyRef = useRef<string>(crypto.randomUUID());
  const [useRefs, setUseRefs] = useState(false);
  const [refScale, setRefScale] = useState(0.6);
  const [loraId, setLoraId] = useState("");
  const [loraScale, setLoraScale] = useState(1);
  const loras = useQuery({ queryKey: ["models", "lora"], queryFn: () => listModels(), select: (all) => all.filter((m) => m.kind !== "base" && ["approved", "awaiting_review"].includes(m.status)) });
  const lora = loras.data?.find((m) => m.id === loraId);
  const qc = useQueryClient();
  const [charId, setCharId] = useState(""); // "" = nhân vật mới / tự do
  const chars = useQuery({ queryKey: ["characters"], queryFn: listCharacters });
  const character = chars.data?.find((c) => c.id === charId);
  const version = character?.current_version ?? null;

  function pickCharacter(id: string) {
    setCharId(id);
    setOverride(null);
    const v = chars.data?.find((c) => c.id === id)?.current_version;
    if (v) {
      setIdentityEn(v.identity.identity_en);
      if (v.identity.description_vi) setOriginal(v.identity.description_vi);
      if (v.identity.negative_prompt) setNegative(v.identity.negative_prompt);
    }
  }
  const setRef = useMutation({
    mutationFn: (asset: string) => addReference(charId, asset),
    onSuccess: () => { qc.invalidateQueries({ queryKey: ["characters"] }); qc.invalidateQueries({ queryKey: ["character", charId] }); },
  });

  // Dùng lại tham số từ Lịch sử: điền lại toàn bộ form theo job đã chọn.
  useEffect(() => {
    if (!reuse) return;
    const { job } = reuse;
    const c = job.config;
    setOriginal(job.prompt_original);
    setOverride(job.prompt_compiled ?? null);
    setNegative(c.negative_prompt ?? "");
    setAspect(c.aspect_ratio ?? "1:1");
    setNum(c.num_images ?? 1);
    setSeed(c.seed ?? 42);
    setSteps(c.steps ?? 28);
    setGuidance(c.guidance_scale ?? 6);
    setCharId(job.character?.id ?? "");
    setUseRefs(!!c.use_references);
    setRefScale(c.reference_scale ?? 0.6);
    setLoraId(c.lora_version_id ?? "");
    setLoraScale(c.lora_scale ?? 1);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [reuse?.n]);

  const compiled = useMemo(() => compilePrompt(identityEn, sel, extraEn), [identityEn, sel, extraEn]);
  const finalPrompt = override ?? compiled;

  const submit = useMutation({
    mutationFn: () => {
      const body: GenerationBody = {
        character_version_id: version?.id ?? null,
        prompt_original: original, prompt_compiled: finalPrompt, negative_prompt: negative,
        aspect_ratio: aspect, num_images: num, seed, steps, guidance_scale: guidance,
        ...(useRefs && version?.references.length ? { use_references: true, reference_scale: refScale } : {}),
        ...(lora ? { lora_version_id: lora.id, lora_scale: loraScale, evaluation: lora.status !== "approved" } : {}),
      };
      return createJob(body, keyRef.current);
    },
    onSuccess: (r) => setJobId(r.job_id),
  });

  // Polling; dừng khi job đạt trạng thái cuối.
  const [tick, setTick] = useState(0);
  const job = useQuery({ queryKey: ["job", jobId, tick], queryFn: () => getJob(jobId!), enabled: !!jobId, retry: false });
  const state = job.data?.state;
  const terminal = state && !["queued", "running"].includes(state);
  useEffect(() => {
    if (!jobId || terminal) return;
    const t = setTimeout(() => setTick((n) => n + 1), 2000);
    return () => clearTimeout(t);
  }, [jobId, tick, terminal]);

  const busy = submit.isPending || (!!jobId && !terminal);
  const ready = original.trim().length > 0 && finalPrompt.trim().length > 0;
  const st = state ? STATE_LABEL[state] ?? { text: state, tone: "" as Tone } : null;
  const wrongLora = !!(lora?.character_id && character && lora.character_id !== character.id);

  function start() {
    keyRef.current = crypto.randomUUID(); // mỗi lần bấm là một job mới
    setJobId(null);
    submit.mutate();
  }

  return (
    <>
      <PageHead title="Tạo ảnh nhân vật" sub="Chọn nhân vật, mô tả cảnh bằng preset, rồi tạo ảnh bằng SDXL. Ảnh được lưu thẳng vào PostgreSQL." />
      <div className="split">
        <div className="stack">
          <Card title="Nhân vật & mô hình" sub="Dùng hồ sơ đã lưu để giữ nhất quán diện mạo.">
            <div className="stack">
              <Field label="Nhân vật" hint={version ? `Version ${version.version} · ${version.references.length} ảnh chuẩn` : "Chọn “tự do” để thử nhân vật mới."}>
                <select value={charId} onChange={(e) => pickCharacter(e.target.value)}>
                  <option value="">Nhân vật mới / tự do</option>
                  {chars.data?.map((c) => <option key={c.id} value={c.id}>{c.code} · {c.name} (v{c.current_version?.version})</option>)}
                </select>
              </Field>

              {version && (
                <div className="card flat" style={{ padding: 12 }}>
                  <div className="stack-sm">
                    <label className="check">
                      <input type="checkbox" checked={useRefs && version.references.length > 0} disabled={version.references.length === 0}
                        onChange={(e) => setUseRefs(e.target.checked)} />
                      Giữ diện mạo bằng ảnh chuẩn <Badge tone="primary">IP-Adapter</Badge>
                    </label>
                    {version.references.length === 0 ? (
                      <small>{character?.code} chưa có ảnh chuẩn — tạo ảnh rồi bấm “Đặt làm ảnh chuẩn”.</small>
                    ) : (
                      <div className="row" style={{ gap: 6 }}>
                        {version.references.slice(0, 4).map((a) => <img key={a} className="thumb" src={thumbUrl(a)} alt="ảnh chuẩn" style={{ opacity: useRefs ? 1 : 0.4 }} />)}
                      </div>
                    )}
                    {useRefs && version.references.length > 0 && (
                      <Field label={`Độ mạnh tham chiếu: ${refScale.toFixed(2)}`} hint="Mạnh quá có thể kéo trang phục/khung hình giống ảnh chuẩn.">
                        <input type="range" min={0} max={1} step={0.05} value={refScale} onChange={(e) => setRefScale(+e.target.value)} />
                      </Field>
                    )}
                  </div>
                </div>
              )}

              <Field label="LoRA">
                <select value={loraId} onChange={(e) => setLoraId(e.target.value)}>
                  <option value="">Không dùng LoRA</option>
                  {loras.data?.map((m) => <option key={m.id} value={m.id}>{m.name} — {m.status === "approved" ? "đã duyệt" : "chờ duyệt (chỉ đánh giá)"}</option>)}
                </select>
              </Field>
              {lora && (
                <div className="row">
                  <small>Trigger <code>{lora.trigger_token}</code> tự thêm vào prompt.</small>
                  <span className="spacer" />
                  <label className="row" style={{ gap: 6 }}><small>Scale</small>
                    <input type="number" step={0.1} min={0} max={2} value={loraScale} onChange={(e) => setLoraScale(+e.target.value)} style={{ width: 70 }} />
                  </label>
                </div>
              )}
              {wrongLora && <Alert tone="bad">LoRA này thuộc nhân vật khác — hệ thống sẽ từ chối.</Alert>}
            </div>
          </Card>

          <Card title="Mô tả & preset" sub="Model hiểu tiếng Anh tốt hơn; preset được ghép thành prompt, không dịch tự do.">
            <div className="stack">
              <Field label="Mô tả (tiếng Việt, lưu làm bản gốc)">
                <textarea value={original} onChange={(e) => setOriginal(e.target.value)} rows={3} />
              </Field>
              <Field label="Nhận dạng nhân vật (tiếng Anh, gửi cho model)">
                <input value={identityEn} onChange={(e) => setIdentityEn(e.target.value)} />
              </Field>
              <div className="cols-2">
                {FIELDS.map((f) => (
                  <Field key={f.key} label={f.label}>
                    <select value={sel[f.key] ?? ""} onChange={(e) => setSel({ ...sel, [f.key]: e.target.value || undefined })}>
                      <option value="">(không chọn)</option>
                      {f.options.map((o) => <option key={o.en} value={o.en}>{o.vi}</option>)}
                    </select>
                  </Field>
                ))}
              </div>
              <Field label="Ghi chú thêm (tiếng Anh)">
                <input value={extraEn} onChange={(e) => setExtraEn(e.target.value)} placeholder="vd: holding a cup of tea" />
              </Field>
            </div>
          </Card>

          <Card title="Prompt gửi model" actions={override !== null && <button className="btn sm ghost" type="button" onClick={() => setOverride(null)}><Icon name="refresh" className="sm" />Đặt lại theo preset</button>}>
            <div className="stack">
              <Field label="Prompt (có thể sửa tay)">
                <textarea className="prompt-box" value={finalPrompt} onChange={(e) => setOverride(e.target.value)} rows={4} />
              </Field>
              <Field label="Negative prompt">
                <input value={negative} onChange={(e) => setNegative(e.target.value)} />
              </Field>
            </div>
          </Card>

          <Card title="Thông số">
            <div className="stack">
              <Field label="Khung hình" hint={aspect === "9:16" && useRefs ? "⚠ 9:16 kèm tham chiếu IP-Adapter rất nặng trên GPU 8GB (đã đo ~28 giây/bước, ~10 phút/ảnh). Nên dùng 768² hoặc 1:1." : "Với 8GB VRAM, 9:16 có thể hết bộ nhớ — thử 1:1 hoặc 768² trước."}>
                <Segmented label="Khung hình" value={aspect} options={ASPECTS} onChange={setAspect} />
              </Field>
              <div className="cols-4">
                <Field label="Số ảnh"><input type="number" min={1} max={4} value={num} onChange={(e) => setNum(+e.target.value)} /></Field>
                <Field label="Seed"><input type="number" min={0} value={seed} onChange={(e) => setSeed(+e.target.value)} /></Field>
                <Field label="Số bước"><input type="number" min={1} max={100} value={steps} onChange={(e) => setSteps(+e.target.value)} /></Field>
                <Field label="Guidance"><input type="number" step={0.5} min={0} max={20} value={guidance} onChange={(e) => setGuidance(+e.target.value)} /></Field>
              </div>
            </div>
          </Card>

          <button className="btn primary lg block" disabled={!ready || busy} onClick={start}>
            {busy ? <><Spinner /> Đang xử lý…</> : <><Icon name="sparkles" className="sm" /> Tạo ảnh</>}
          </button>
          {submit.error && <Alert tone="bad">Lỗi: {(submit.error as Error).message}</Alert>}
        </div>

        <div className="sticky stack">
          <Card title="Kết quả" sub={jobId ? <>Job <code>{jobId.slice(0, 8)}</code></> : undefined}
            actions={st && <Badge tone={st.tone}>{(state === "queued" || state === "running") && <Spinner />}{st.text}</Badge>}>
            {!jobId && (
              <Empty icon="image" title="Chưa có ảnh nào">Điền mô tả rồi bấm “Tạo ảnh”. Lần đầu worker nạp model nên có thể mất 1–2 phút.</Empty>
            )}
            {jobId && !terminal && (
              <div className="stack-sm" style={{ marginBottom: 14 }}>
                <div className="progress indeterminate" aria-label="Đang tạo"><i /></div>
                <small>Tiến độ chỉ hiển thị trạng thái thật của job (không bịa phần trăm).</small>
              </div>
            )}
            {state === "failed" && <Alert tone="bad"><strong>Tạo ảnh thất bại.</strong> {job.data?.error}</Alert>}
            {job.error && <Alert tone="bad">Không đọc được job: {(job.error as Error).message}</Alert>}
            {setRef.error && <Alert tone="bad">Lỗi: {(setRef.error as Error).message}</Alert>}
            {!!job.data?.outputs.length && (
              <div className="gallery big">
                {job.data.outputs.map((o) => (
                  <figure key={o.asset_id} className="shot" style={{ margin: 0 }}>
                    <img src={assetUrl(o.asset_id)} alt={`Ảnh seed ${o.seed}`} />
                    <figcaption className="shot-bar">
                      <Badge>seed {o.seed}</Badge>
                      <span className="spacer" />
                      <a className="btn sm" href={assetUrl(o.asset_id)} download={`emvoox-${o.asset_id.slice(0, 8)}.png`}><Icon name="download" className="sm" />Tải</a>
                      {charId && (version?.references.includes(o.asset_id)
                        ? <Badge tone="ok"><Icon name="check" className="sm" />Ảnh chuẩn</Badge>
                        : <button className="btn sm" disabled={setRef.isPending} onClick={() => setRef.mutate(o.asset_id)}><Icon name="star" className="sm" />Đặt làm ảnh chuẩn</button>)}
                    </figcaption>
                  </figure>
                ))}
              </div>
            )}
          </Card>
        </div>
      </div>
    </>
  );
}
