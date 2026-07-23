//! R7 (native reference) — the value-transfer state transition + post-state root.
//!
//! Given committed block data (an EIP-1559 transfer tx + the touched pre-state
//! accounts + block env), applies ev-reth's 1559 mechanics and returns the
//! post-state accounts and `post_state_root` (via the native MPT). This is the
//! golden the assembled in-circuit STF must reproduce; it is differential-tested
//! against ev-reth's `evm-oracle` (post balances/nonces) and its alloy-trie
//! state root.
//!
//! Mechanics (matches the evm-oracle transfer example):
//!   effective_gas_price = base_fee + min(max_priority_fee, max_fee - base_fee)
//!   gas_used            = 21000 (pure value transfer)
//!   sender.balance     -= value + gas_used * effective_gas_price
//!   sender.nonce       += 1
//!   recipient.balance  += value
//!   coinbase.balance   += gas_used * (effective_gas_price - base_fee)   // tip
//!   (base_fee * gas_used is burned)

use crate::mpt::{state_root, Account};
use crate::rlp::Eip1559Tx;
use num_bigint::BigInt;

pub const GAS_TRANSFER: u64 = 21000;

#[derive(Clone, Debug)]
pub struct BlockEnv {
    pub base_fee: BigInt,
    pub coinbase: [u8; 20],
}

#[derive(Clone, Debug)]
pub struct TransferBlock {
    pub pre: Vec<Account>, // pre-state accounts (sender, recipient, coinbase, ...)
    pub tx: Eip1559Tx,
    pub sender: [u8; 20], // recovered via ecrecover
    pub env: BlockEnv,
}

fn find<'a>(accts: &'a mut [Account], addr: &[u8; 20]) -> Option<&'a mut Account> {
    accts.iter_mut().find(|a| &a.address == addr)
}

/// Apply the transfer; returns (post_accounts, post_state_root). Errors on a
/// nonce mismatch or insufficient balance (tx rejected — state unchanged root).
pub fn apply(block: &TransferBlock) -> Result<(Vec<Account>, [u8; 32]), String> {
    let mut post = block.pre.clone();
    let priority = {
        let head = &block.tx.max_fee - &block.env.base_fee;
        if block.tx.max_priority_fee < head { block.tx.max_priority_fee.clone() } else { head }
    };
    let effective = &block.env.base_fee + &priority;
    let gas_cost = BigInt::from(GAS_TRANSFER) * &effective;
    let tip = BigInt::from(GAS_TRANSFER) * &priority;
    let total_debit = &block.tx.value + &gas_cost;

    // sender
    {
        let s = find(&mut post, &block.sender).ok_or("sender not in pre-state")?;
        if s.nonce != block.tx.nonce {
            return Err(format!("nonce mismatch: state {} tx {}", s.nonce, block.tx.nonce));
        }
        if s.balance < total_debit {
            return Err("insufficient balance".into());
        }
        s.balance -= &total_debit;
        s.nonce += 1;
    }
    // recipient
    {
        let r = find(&mut post, &block.tx.to).ok_or("recipient not in pre-state")?;
        r.balance += &block.tx.value;
    }
    // coinbase (tip)
    {
        let c = find(&mut post, &block.env.coinbase).ok_or("coinbase not in pre-state")?;
        c.balance += &tip;
    }
    let root = state_root(&post);
    Ok((post, root))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mpt::Account;
    use crate::rlp::Eip1559Tx;
    use num_bigint::BigInt;

    fn hx(b: &[u8]) -> String {
        b.iter().map(|x| format!("{:02x}", x)).collect()
    }

    #[test]
    fn transfer_transition_is_consistent() {
        let sender = [0x11u8; 20];
        let recipient = [0xABu8; 20];
        let coinbase = [0xCCu8; 20];
        let pre = vec![
            Account::eoa(sender, 7, BigInt::from(1_000_000_000_000_000_000u64)),
            Account::eoa(recipient, 0, BigInt::from(0u64)),
            Account::eoa(coinbase, 0, BigInt::from(0u64)),
        ];
        let tx = Eip1559Tx {
            chain_id: 1, nonce: 7,
            max_priority_fee: BigInt::from(2u64),
            max_fee: BigInt::from(20u64),
            gas_limit: 21000, to: recipient,
            value: BigInt::from(1_000_000_000_000_000u64),
            data: vec![], y_parity: 0, r: BigInt::from(1), s: BigInt::from(1),
        };
        let env = BlockEnv { base_fee: BigInt::from(7u64), coinbase };
        let block = TransferBlock { pre: pre.clone(), tx, sender, env };
        let pre_root = state_root(&pre);
        let (post, root) = apply(&block).unwrap();

        let s = post.iter().find(|a| a.address == sender).unwrap();
        let r = post.iter().find(|a| a.address == recipient).unwrap();
        let c = post.iter().find(|a| a.address == coinbase).unwrap();
        // effective = 7 + min(2, 20-7)=9; gas=21000*9=189000; tip=21000*2=42000.
        assert_eq!(s.balance, BigInt::from(1_000_000_000_000_000_000u64) - BigInt::from(1_000_000_000_000_000u64) - BigInt::from(189_000u64));
        assert_eq!(s.nonce, 8);
        assert_eq!(r.balance, BigInt::from(1_000_000_000_000_000u64));
        assert_eq!(c.balance, BigInt::from(42_000u64));
        assert_ne!(pre_root, root, "state root must change");
        println!("R7 transfer STF: pre_root=0x{} post_root=0x{}", hx(&pre_root), hx(&root));
    }
}
