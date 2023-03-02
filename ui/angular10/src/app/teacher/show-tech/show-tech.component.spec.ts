import { ComponentFixture, TestBed } from '@angular/core/testing';

import { ShowTechComponent } from './show-tech.component';

describe('ShowTechComponent', () => {
  let component: ShowTechComponent;
  let fixture: ComponentFixture<ShowTechComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [ ShowTechComponent ]
    })
    .compileComponents();

    fixture = TestBed.createComponent(ShowTechComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
