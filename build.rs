//! Stamps the git commit into the extension's `__build__` string
//! (`<version>+g<sha12>`, `-dirty` appended when the tree has uncommitted
//! changes) so mined-artifact provenance pins the exact code that produced it.
//! Outside a git checkout (sdist builds) the stamp is the bare version.

use std::process::Command;

fn git(args: &[&str]) -> Option<String> {
    let out = Command::new("git").args(args).output().ok()?;
    if !out.status.success() {
        return None;
    }
    Some(String::from_utf8_lossy(&out.stdout).trim().to_string())
}

fn main() {
    let version = std::env::var("CARGO_PKG_VERSION").unwrap();
    let build = match git(&["rev-parse", "--short=12", "HEAD"]) {
        Some(sha) => {
            // -uno: only tracked-file changes make a build dirty; untracked clutter
            // (local asset checkouts, scratch dirs) doesn't alter what compiles.
            let dirty = git(&["status", "--porcelain", "-uno"]).is_some_and(|s| !s.is_empty());
            format!("{version}+g{sha}{}", if dirty { "-dirty" } else { "" })
        }
        None => version,
    };
    println!("cargo:rustc-env=SIMPLIPY_BUILD={build}");
    // HEAD moves on branch switches, the index on staging/commits; watching both keeps
    // the stamp current without rebuilding on unrelated file edits.
    println!("cargo:rerun-if-changed=.git/HEAD");
    println!("cargo:rerun-if-changed=.git/index");
}
