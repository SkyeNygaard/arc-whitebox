#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <gmp.h>
/* Minimal MPFR ABI declarations; the runtime library is present but development headers are not. */
typedef long int mpfr_prec_t;
typedef long int mpfr_exp_t;
typedef int mpfr_sign_t;
typedef enum { MPFR_RNDN=0, MPFR_RNDZ=1, MPFR_RNDU=2, MPFR_RNDD=3, MPFR_RNDA=4, MPFR_RNDF=5, MPFR_RNDNA=-1 } mpfr_rnd_t;
typedef struct { mpfr_prec_t _mpfr_prec; mpfr_sign_t _mpfr_sign; mpfr_exp_t _mpfr_exp; mp_limb_t *_mpfr_d; } __mpfr_struct;
typedef __mpfr_struct mpfr_t[1];
typedef __mpfr_struct *mpfr_ptr;
typedef const __mpfr_struct *mpfr_srcptr;
void mpfr_set_default_prec(mpfr_prec_t);
void mpfr_init2(mpfr_ptr, mpfr_prec_t);
void mpfr_clear(mpfr_ptr);
int mpfr_set(mpfr_ptr, mpfr_srcptr, mpfr_rnd_t);
int mpfr_set_ui(mpfr_ptr, unsigned long, mpfr_rnd_t);
int mpfr_set_si(mpfr_ptr, long, mpfr_rnd_t);
int mpfr_set_str(mpfr_ptr, const char*, int, mpfr_rnd_t);
int mpfr_set_q(mpfr_ptr, mpq_srcptr, mpfr_rnd_t);
int mpfr_add(mpfr_ptr, mpfr_srcptr, mpfr_srcptr, mpfr_rnd_t);
int mpfr_sub(mpfr_ptr, mpfr_srcptr, mpfr_srcptr, mpfr_rnd_t);
int mpfr_mul(mpfr_ptr, mpfr_srcptr, mpfr_srcptr, mpfr_rnd_t);
int mpfr_div(mpfr_ptr, mpfr_srcptr, mpfr_srcptr, mpfr_rnd_t);
int mpfr_div_ui(mpfr_ptr, mpfr_srcptr, unsigned long, mpfr_rnd_t);
int mpfr_ui_div(mpfr_ptr, unsigned long, mpfr_srcptr, mpfr_rnd_t);
int mpfr_mul_ui(mpfr_ptr, mpfr_srcptr, unsigned long, mpfr_rnd_t);
int mpfr_sub_ui(mpfr_ptr, mpfr_srcptr, unsigned long, mpfr_rnd_t);
int mpfr_neg(mpfr_ptr, mpfr_srcptr, mpfr_rnd_t);
int mpfr_abs(mpfr_ptr, mpfr_srcptr, mpfr_rnd_t);
int mpfr_sqrt(mpfr_ptr, mpfr_srcptr, mpfr_rnd_t);
int mpfr_asin(mpfr_ptr, mpfr_srcptr, mpfr_rnd_t);
int mpfr_const_pi(mpfr_ptr, mpfr_rnd_t);
int mpfr_cmp(mpfr_srcptr, mpfr_srcptr);
int mpfr_cmp_si(mpfr_srcptr, long);
int mpfr_cmp_ui(mpfr_srcptr, unsigned long);
int mpfr_sgn(mpfr_srcptr);
void mpfr_set_inf(mpfr_ptr, int);
int __gmpfr_out_str(FILE*, int, size_t, mpfr_srcptr, mpfr_rnd_t);
#define mpfr_out_str __gmpfr_out_str
int mpfr_asprintf(char**, const char*, ...);
void mpfr_free_str(char*);
const char *mpfr_get_version(void);

#define PREC 256
#define MAX_LINE 8192
#define MAX_DEPTH 12

typedef struct { mpfr_t lo, hi; } I;
static void ii(I *x){ mpfr_init2(x->lo,PREC); mpfr_init2(x->hi,PREC); }
static void ci(I *x){ mpfr_clear(x->lo); mpfr_clear(x->hi); }
static void set_ui_i(I *x,unsigned long n){mpfr_set_ui(x->lo,n,MPFR_RNDD);mpfr_set_ui(x->hi,n,MPFR_RNDU);} 
static void set_q_i(I *x,const mpq_t q){mpfr_set_q(x->lo,q,MPFR_RNDD);mpfr_set_q(x->hi,q,MPFR_RNDU);} 
static void copy_i(I *r,const I *a){mpfr_set(r->lo,a->lo,MPFR_RNDD);mpfr_set(r->hi,a->hi,MPFR_RNDU);} 
static void add_i(I *r,const I*a,const I*b){mpfr_add(r->lo,a->lo,b->lo,MPFR_RNDD);mpfr_add(r->hi,a->hi,b->hi,MPFR_RNDU);} 
static void neg_i(I*r,const I*a){mpfr_neg(r->lo,a->hi,MPFR_RNDD);mpfr_neg(r->hi,a->lo,MPFR_RNDU);} 
static void sub_i(I*r,const I*a,const I*b){I nb;ii(&nb);neg_i(&nb,b);add_i(r,a,&nb);ci(&nb);} 
static void mul_i(I*r,const I*a,const I*b){
 mpfr_t dl[4],du[4]; for(int k=0;k<4;k++){mpfr_init2(dl[k],PREC);mpfr_init2(du[k],PREC);} 
 mpfr_mul(dl[0],a->lo,b->lo,MPFR_RNDD); mpfr_mul(du[0],a->lo,b->lo,MPFR_RNDU);
 mpfr_mul(dl[1],a->lo,b->hi,MPFR_RNDD); mpfr_mul(du[1],a->lo,b->hi,MPFR_RNDU);
 mpfr_mul(dl[2],a->hi,b->lo,MPFR_RNDD); mpfr_mul(du[2],a->hi,b->lo,MPFR_RNDU);
 mpfr_mul(dl[3],a->hi,b->hi,MPFR_RNDD); mpfr_mul(du[3],a->hi,b->hi,MPFR_RNDU);
 mpfr_set(r->lo,dl[0],MPFR_RNDD);mpfr_set(r->hi,du[0],MPFR_RNDU);
 for(int k=1;k<4;k++){if(mpfr_cmp(dl[k],r->lo)<0)mpfr_set(r->lo,dl[k],MPFR_RNDD);if(mpfr_cmp(du[k],r->hi)>0)mpfr_set(r->hi,du[k],MPFR_RNDU);} 
 for(int k=0;k<4;k++){mpfr_clear(dl[k]);mpfr_clear(du[k]);}
}
static void div_i(I*r,const I*a,const I*b){
 if(mpfr_sgn(b->lo)<=0 && mpfr_sgn(b->hi)>=0){fprintf(stderr,"division by interval containing zero\n");exit(3);} 
 I inv;ii(&inv);mpfr_ui_div(inv.lo,1,b->hi,MPFR_RNDD);mpfr_ui_div(inv.hi,1,b->lo,MPFR_RNDU);mul_i(r,a,&inv);ci(&inv);
}
static void sqrt_i(I*r,const I*a){if(mpfr_sgn(a->lo)<0){fprintf(stderr,"sqrt negative\n");exit(4);}mpfr_sqrt(r->lo,a->lo,MPFR_RNDD);mpfr_sqrt(r->hi,a->hi,MPFR_RNDU);} 
static void asin_i(I*r,const I*a){mpfr_asin(r->lo,a->lo,MPFR_RNDD);mpfr_asin(r->hi,a->hi,MPFR_RNDU);} 
static void abs_mpfr(mpfr_t r,const mpfr_t x,mpfr_rnd_t rnd){mpfr_abs(r,x,rnd);} 

static I PI;

static void kappa_point(I *k,I *kp,const mpfr_t x){
 if(mpfr_cmp_si(x,-1)==0){set_ui_i(k,0);set_ui_i(kp,0);return;} if(mpfr_cmp_ui(x,1)==0){set_ui_i(k,1);set_ui_i(kp,1);return;}
 I X,one,x2,omx2,root,ax,half,pi2,angle,prod,num;ii(&X);ii(&one);ii(&x2);ii(&omx2);ii(&root);ii(&ax);ii(&half);ii(&pi2);ii(&angle);ii(&prod);ii(&num);
 mpfr_set(X.lo,x,MPFR_RNDD);mpfr_set(X.hi,x,MPFR_RNDU);set_ui_i(&one,1);
 mul_i(&x2,&X,&X);sub_i(&omx2,&one,&x2);sqrt_i(&root,&omx2);asin_i(&ax,&X);
 mpfr_set_ui(half.lo,1,MPFR_RNDD);mpfr_div_ui(half.lo,half.lo,2,MPFR_RNDD);mpfr_set_ui(half.hi,1,MPFR_RNDU);mpfr_div_ui(half.hi,half.hi,2,MPFR_RNDU);
 // pi/2 computed directly
 mpfr_div_ui(pi2.lo,PI.lo,2,MPFR_RNDD);mpfr_div_ui(pi2.hi,PI.hi,2,MPFR_RNDU);
 add_i(&angle,&pi2,&ax);mul_i(&prod,&angle,&X);add_i(&num,&root,&prod);div_i(k,&num,&PI);div_i(&prod,&ax,&PI);add_i(kp,&half,&prod);
 ci(&X);ci(&one);ci(&x2);ci(&omx2);ci(&root);ci(&ax);ci(&half);ci(&pi2);ci(&angle);ci(&prod);ci(&num);
}

static void kappa_interval(I *k,I *kp,const I *x){
 I kl,kpl,kh,kph;ii(&kl);ii(&kpl);ii(&kh);ii(&kph);kappa_point(&kl,&kpl,x->lo);kappa_point(&kh,&kph,x->hi);
 mpfr_set(k->lo,kl.lo,MPFR_RNDD);mpfr_set(k->hi,kh.hi,MPFR_RNDU);mpfr_set(kp->lo,kpl.lo,MPFR_RNDD);mpfr_set(kp->hi,kph.hi,MPFR_RNDU);
 ci(&kl);ci(&kpl);ci(&kh);ci(&kph);
}

static void kappa_second(I *r,const I*x){
 mpfr_t alo,ahi,maxabs,minabs;mpfr_init2(alo,PREC);mpfr_init2(ahi,PREC);mpfr_init2(maxabs,PREC);mpfr_init2(minabs,PREC);
 abs_mpfr(alo,x->lo,MPFR_RNDU);abs_mpfr(ahi,x->hi,MPFR_RNDU);if(mpfr_cmp(alo,ahi)>=0)mpfr_set(maxabs,alo,MPFR_RNDU);else mpfr_set(maxabs,ahi,MPFR_RNDU);
 if(mpfr_sgn(x->lo)<=0 && mpfr_sgn(x->hi)>=0)mpfr_set_ui(minabs,0,MPFR_RNDD);else {if(mpfr_cmp(alo,ahi)<=0)mpfr_set(minabs,alo,MPFR_RNDD);else mpfr_set(minabs,ahi,MPFR_RNDD);} 
 I one,minA,maxA,minSq,maxSq,minArg,maxArg,smin,smax,den;ii(&one);ii(&minA);ii(&maxA);ii(&minSq);ii(&maxSq);ii(&minArg);ii(&maxArg);ii(&smin);ii(&smax);ii(&den);set_ui_i(&one,1);
 mpfr_set(minA.lo,minabs,MPFR_RNDD);mpfr_set(minA.hi,minabs,MPFR_RNDU);mpfr_set(maxA.lo,maxabs,MPFR_RNDD);mpfr_set(maxA.hi,maxabs,MPFR_RNDU);
 mul_i(&minSq,&minA,&minA);mul_i(&maxSq,&maxA,&maxA);sub_i(&minArg,&one,&maxSq);sub_i(&maxArg,&one,&minSq);sqrt_i(&smin,&minArg);sqrt_i(&smax,&maxArg);
 mpfr_mul(den.lo,PI.lo,smin.lo,MPFR_RNDD);mpfr_mul(den.hi,PI.hi,smax.hi,MPFR_RNDU);I oneI;ii(&oneI);set_ui_i(&oneI,1);div_i(r,&oneI,&den);ci(&oneI);
 ci(&one);ci(&minA);ci(&maxA);ci(&minSq);ci(&maxSq);ci(&minArg);ci(&maxArg);ci(&smin);ci(&smax);ci(&den);mpfr_clear(alo);mpfr_clear(ahi);mpfr_clear(maxabs);mpfr_clear(minabs);
}

static void deep_second(I*r,const mpq_t aq,const mpq_t bq){
 I z,p,q;ii(&z);ii(&p);ii(&q);mpfr_set_q(z.lo,aq,MPFR_RNDD);mpfr_set_q(z.hi,bq,MPFR_RNDU);set_ui_i(&p,1);set_ui_i(&q,0);
 for(int it=0;it<32;it++){
  I kp2,znew,kp,p2,t1,t2,qnew,pnew;ii(&kp2);ii(&znew);ii(&kp);ii(&p2);ii(&t1);ii(&t2);ii(&qnew);ii(&pnew);
  kappa_second(&kp2,&z);kappa_interval(&znew,&kp,&z);mul_i(&p2,&p,&p);mul_i(&t1,&kp2,&p2);mul_i(&t2,&kp,&q);add_i(&qnew,&t1,&t2);mul_i(&pnew,&kp,&p);
  copy_i(&z,&znew);copy_i(&p,&pnew);copy_i(&q,&qnew);
  ci(&kp2);ci(&znew);ci(&kp);ci(&p2);ci(&t1);ci(&t2);ci(&qnew);ci(&pnew);
 }
 copy_i(r,&q);ci(&z);ci(&p);ci(&q);
}


static void load_qs(const char*path,mpq_t*q,int n){FILE*f=fopen(path,"r");if(!f){perror(path);exit(2);}char line[MAX_LINE];for(int i=0;i<n;i++){if(!fgets(line,sizeof line,f)){fprintf(stderr,"coeff missing\n");exit(2);}line[strcspn(line,"\r\n")]=0;mpq_init(q[i]);if(mpq_set_str(q[i],line,10)!=0){fprintf(stderr,"bad coeff\n");exit(2);}mpq_canonicalize(q[i]);}fclose(f);}
static void deep_k(I*k,const mpq_t x){I z,p;ii(&z);ii(&p);mpfr_set_q(z.lo,x,MPFR_RNDD);mpfr_set_q(z.hi,x,MPFR_RNDU);set_ui_i(&p,1);for(int it=0;it<32;it++){I zn,kp,pn;ii(&zn);ii(&kp);ii(&pn);kappa_interval(&zn,&kp,&z);mul_i(&pn,&p,&kp);copy_i(&z,&zn);copy_i(&p,&pn);ci(&zn);ci(&kp);ci(&pn);}copy_i(k,&z);ci(&z);ci(&p);}
static char* ms(const mpfr_t x){char*s=NULL;mpfr_asprintf(&s,"%.120Re",x);return s;}
int main(int argc,char**argv){if(argc!=4){fprintf(stderr,"usage: %s coeff.tsv kernel_mean.json output.json\n",argv[0]);return 2;}mpfr_set_default_prec(512);ii(&PI);mpfr_const_pi(PI.lo,MPFR_RNDD);mpfr_const_pi(PI.hi,MPFR_RNDU);mpq_t c[6];load_qs(argv[1],c,6);mpq_t c0,h1,oneq,boundq,tmpq;mpq_init(c0);mpq_init(h1);mpq_init(oneq);mpq_init(boundq);mpq_init(tmpq);mpq_set(c0,c[0]);mpq_set_ui(h1,0,1);for(int i=0;i<6;i++)mpq_add(h1,h1,c[i]);mpq_set_ui(oneq,1,1);mpq_sub(tmpq,oneq,h1);mpq_div_2exp(tmpq,tmpq,0);mpq_t nq;mpq_init(nq);mpq_set_ui(nq,66048,1);mpq_div(tmpq,tmpq,nq);mpq_add(boundq,c0,tmpq);
 mpq_t q[5];for(int i=0;i<5;i++)mpq_init(q[i]);mpq_set_si(q[0],-1,1);mpq_set_si(q[1],-1,16);mpq_set_si(q[2],0,1);mpq_set_si(q[3],1,16);mpq_set_si(q[4],1,1);I K[5];for(int i=0;i<5;i++){ii(&K[i]);deep_k(&K[i],q[i]);}
 I row,t,sum,energy,bound;ii(&row);ii(&t);ii(&sum);ii(&energy);ii(&bound);add_i(&row,&K[4],&K[0]);I w510,w32768,wN;ii(&w510);ii(&w32768);ii(&wN);set_ui_i(&w510,510);set_ui_i(&w32768,32768);set_ui_i(&wN,66048);mul_i(&t,&K[2],&w510);add_i(&sum,&row,&t);copy_i(&row,&sum);I pair;ii(&pair);add_i(&pair,&K[1],&K[3]);mul_i(&t,&pair,&w32768);add_i(&sum,&row,&t);copy_i(&row,&sum);div_i(&energy,&row,&wN);set_q_i(&bound,boundq);
 /* parse A0 values from the known JSON using a tiny line scanner */
 FILE*mf=fopen(argv[2],"r");if(!mf){perror(argv[2]);return 2;}char line[16384],al[8192]={0},au[8192]={0};while(fgets(line,sizeof line,mf)){char*p;if((p=strstr(line,"\"A0_lower\""))){p=strchr(p,':');p=strchr(p,'\"');char*e=strchr(p+1,'\"');memcpy(al,p+1,e-p-1);al[e-p-1]=0;}if((p=strstr(line,"\"A0_upper\""))){p=strchr(p,':');p=strchr(p,'\"');char*e=strchr(p+1,'\"');memcpy(au,p+1,e-p-1);au[e-p-1]=0;}}fclose(mf);if(!al[0]||!au[0]){fprintf(stderr,"A0 parse fail\n");return 2;}I A0;ii(&A0);mpfr_set_str(A0.lo,al,10,MPFR_RNDD);mpfr_set_str(A0.hi,au,10,MPFR_RNDU);
 mpfr_t kmse_lo,kmse_hi,opt_lo,ratio,excess,percent,addgap;mpfr_init2(kmse_lo,512);mpfr_init2(kmse_hi,512);mpfr_init2(opt_lo,512);mpfr_init2(ratio,512);mpfr_init2(excess,512);mpfr_init2(percent,512);mpfr_init2(addgap,512);mpfr_sub(kmse_lo,energy.lo,A0.hi,MPFR_RNDD);mpfr_sub(kmse_hi,energy.hi,A0.lo,MPFR_RNDU);mpfr_sub(opt_lo,bound.lo,A0.hi,MPFR_RNDD);if(mpfr_sgn(opt_lo)<=0){fprintf(stderr,"nonpositive optimum lower\n");return 5;}mpfr_div(ratio,kmse_hi,opt_lo,MPFR_RNDU);mpfr_sub_ui(excess,ratio,1,MPFR_RNDU);mpfr_mul_ui(percent,excess,100,MPFR_RNDU);mpfr_sub(addgap,energy.hi,bound.lo,MPFR_RNDU);
 char*el=ms(energy.lo),*eh=ms(energy.hi),*bl=ms(bound.lo),*bh=ms(bound.hi),*ml=ms(kmse_lo),*mh=ms(kmse_hi),*ol=ms(opt_lo),*ra=ms(ratio),*pe=ms(percent),*ag=ms(addgap);FILE*o=fopen(argv[3],"w");fprintf(o,"{\n  \"engine\": \"GMP exact rationals plus MPFR %s direct C\",\n  \"precision_bits\": 512,\n  \"kerdock_energy_lower\": \"%s\",\n  \"kerdock_energy_upper\": \"%s\",\n  \"universal_energy_bound_lower\": \"%s\",\n  \"universal_energy_bound_upper\": \"%s\",\n  \"kerdock_mse_lower\": \"%s\",\n  \"kerdock_mse_upper\": \"%s\",\n  \"optimum_mse_lower\": \"%s\",\n  \"ratio_upper\": \"%s\",\n  \"relative_excess_percent_upper\": \"%s\",\n  \"additive_suboptimality_upper\": \"%s\",\n  \"passed\": true\n}\n",mpfr_get_version(),el,eh,bl,bh,ml,mh,ol,ra,pe,ag);fclose(o);printf("ratio=%s percent=%s\n",ra,pe);
 mpfr_free_str(el);mpfr_free_str(eh);mpfr_free_str(bl);mpfr_free_str(bh);mpfr_free_str(ml);mpfr_free_str(mh);mpfr_free_str(ol);mpfr_free_str(ra);mpfr_free_str(pe);mpfr_free_str(ag);for(int i=0;i<5;i++){ci(&K[i]);mpq_clear(q[i]);}for(int i=0;i<6;i++)mpq_clear(c[i]);mpq_clear(c0);mpq_clear(h1);mpq_clear(oneq);mpq_clear(boundq);mpq_clear(tmpq);mpq_clear(nq);ci(&row);ci(&t);ci(&sum);ci(&energy);ci(&bound);ci(&w510);ci(&w32768);ci(&wN);ci(&pair);ci(&A0);mpfr_clear(kmse_lo);mpfr_clear(kmse_hi);mpfr_clear(opt_lo);mpfr_clear(ratio);mpfr_clear(excess);mpfr_clear(percent);mpfr_clear(addgap);ci(&PI);return 0;}
