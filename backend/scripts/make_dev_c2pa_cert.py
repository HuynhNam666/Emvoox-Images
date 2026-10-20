"""Tạo chứng thư THỬ NGHIỆM để ký C2PA (ES256). Xác minh sẽ cho trạng thái "Valid" nhưng KHÔNG "Trusted":
trình xác minh công cộng chỉ tin chứng thư từ danh sách tin cậy C2PA. Phát hành thật phải dùng chứng thư do CA được C2PA công nhận cấp.

  python scripts/make_dev_c2pa_cert.py --out ./data/c2pa
  # rồi đặt: C2PA_CERT_PATH=./data/c2pa/chain.pem  C2PA_KEY_PATH=./data/c2pa/key.pem
"""
import argparse
import datetime as dt
from pathlib import Path

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID


def name(cn: str) -> x509.Name:
    return x509.Name([x509.NameAttribute(NameOID.ORGANIZATION_NAME, "EMVOOX Dev"), x509.NameAttribute(NameOID.COMMON_NAME, cn)])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="./data/c2pa")
    ap.add_argument("--days", type=int, default=365)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    now = dt.datetime.now(dt.timezone.utc) - dt.timedelta(minutes=5)

    ca_key = ec.generate_private_key(ec.SECP256R1())
    ca = (x509.CertificateBuilder().subject_name(name("EMVOOX Dev CA")).issuer_name(name("EMVOOX Dev CA"))
          .public_key(ca_key.public_key()).serial_number(x509.random_serial_number())
          .not_valid_before(now).not_valid_after(now + dt.timedelta(days=a.days * 2))
          .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
          .add_extension(x509.KeyUsage(digital_signature=False, content_commitment=False, key_encipherment=False, data_encipherment=False,
                                       key_agreement=False, key_cert_sign=True, crl_sign=True, encipher_only=False, decipher_only=False), critical=True)
          .sign(ca_key, hashes.SHA256()))

    key = ec.generate_private_key(ec.SECP256R1())
    leaf = (x509.CertificateBuilder().subject_name(name("EMVOOX Dev Signer")).issuer_name(ca.subject)
            .public_key(key.public_key()).serial_number(x509.random_serial_number())
            .not_valid_before(now).not_valid_after(now + dt.timedelta(days=a.days))
            .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
            .add_extension(x509.KeyUsage(digital_signature=True, content_commitment=False, key_encipherment=False, data_encipherment=False,
                                         key_agreement=False, key_cert_sign=False, crl_sign=False, encipher_only=False, decipher_only=False), critical=True)
            .add_extension(x509.ExtendedKeyUsage([ExtendedKeyUsageOID.EMAIL_PROTECTION]), critical=False)
            .add_extension(x509.SubjectKeyIdentifier.from_public_key(key.public_key()), critical=False)
            .add_extension(x509.AuthorityKeyIdentifier.from_issuer_public_key(ca_key.public_key()), critical=False)
            .sign(ca_key, hashes.SHA256()))

    pem = serialization.Encoding.PEM
    (out / "chain.pem").write_bytes(leaf.public_bytes(pem) + ca.public_bytes(pem))
    (out / "key.pem").write_bytes(key.private_bytes(pem, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()))
    print(f"Đã tạo {out/'chain.pem'} và {out/'key.pem'} (chứng thư THỬ NGHIỆM, không được tin cậy công cộng)")


if __name__ == "__main__":
    main()
